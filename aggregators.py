"""Aggregation backends for federated learning experiments."""
from __future__ import annotations

import os
from collections import OrderedDict
from typing import Dict, Iterable, List

import numpy as np
import torch

try:
    from concrete import fhe
except ImportError as exc:  # pragma: no cover - optional dependency
    fhe = None  # type: ignore
    _CONCRETE_IMPORT_ERROR = exc
else:  # pragma: no cover - difficult to trigger in tests
    _CONCRETE_IMPORT_ERROR = None

try:
    from openfhe import (
        CCParamsCKKSRNS,
        GenCryptoContext,
        PKESchemeFeature,
        SecurityLevel,
    )
except ImportError as exc:  # pragma: no cover - optional dependency
    CCParamsCKKSRNS = None  # type: ignore
    GenCryptoContext = None  # type: ignore
    PKESchemeFeature = None  # type: ignore
    SecurityLevel = None  # type: ignore
    _OPENFHE_IMPORT_ERROR = exc
else:  # pragma: no cover - difficult to trigger in tests
    _OPENFHE_IMPORT_ERROR = None


class AggregationError(RuntimeError):
    """Raised when an aggregation backend fails."""


class BaseModelAggregator:
    """Common utilities for aggregation strategies."""

    def __init__(self, model_structure: torch.nn.Module, num_clients: int, verbose: bool = False) -> None:
        self.num_clients = num_clients
        self.verbose = verbose
        self.weight_shapes: Dict[str, torch.Size] = {}
        self.layer_order: List[str] = []

        for name, param in model_structure.named_parameters():
            self.layer_order.append(name)
            self.weight_shapes[name] = param.shape

    def aggregate(self, client_weights_list: Iterable[OrderedDict[str, torch.Tensor]]) -> OrderedDict[str, torch.Tensor]:
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Helper utilities
    # ------------------------------------------------------------------
    @staticmethod
    def _ensure_cpu_tensor(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.device.type != "cpu":
            return tensor.detach().cpu()
        return tensor.detach().clone()


class PlaintextAggregator(BaseModelAggregator):
    """Performs standard plaintext averaging of model weights."""

    def aggregate(self, client_weights_list: Iterable[OrderedDict[str, torch.Tensor]]) -> OrderedDict[str, torch.Tensor]:
        client_weights_list = list(client_weights_list)
        if not client_weights_list:
            raise AggregationError("No client weights provided for aggregation.")

        aggregated: "OrderedDict[str, torch.Tensor]" = OrderedDict()
        for layer in self.layer_order:
            stacked = torch.stack(
                [self._ensure_cpu_tensor(weights[layer]).float() for weights in client_weights_list],
                dim=0,
            )
            aggregated[layer] = torch.mean(stacked, dim=0)
        return aggregated


class TFHEModelAggregator(BaseModelAggregator):
    """Federated model aggregation using Concrete-ML's TFHE backend."""

    def __init__(
        self,
        model_structure: torch.nn.Module,
        num_clients: int,
        scale_factor: int = 100,
        max_value: int = 50,
        verbose: bool = False,
    ) -> None:
        if fhe is None:  # pragma: no cover - optional dependency
            raise ImportError(
                "concrete-ml is required for TFHE aggregation but is not installed."
            ) from _CONCRETE_IMPORT_ERROR

        super().__init__(model_structure, num_clients, verbose=verbose)
        self.scale_factor = scale_factor
        self.max_value = max_value
        self.circuits: Dict[str, fhe.Compiler] = {}
        self._compile_fhe_circuits()

    # ------------------------------------------------------------------
    # Compilation utilities
    # ------------------------------------------------------------------
    def _compile_fhe_circuits(self) -> None:
        if self.verbose:
            print("🔒 Compiling FHE circuits for model aggregation...")

        for layer_name, shape in self.weight_shapes.items():
            total_elements = int(np.prod(shape))
            if self.verbose:
                print(
                    f"  Compiling circuit for {layer_name} (shape: {shape}, elements: {total_elements})"
                )

            @fhe.compiler({"weights_matrix": "encrypted"})  # type: ignore[misc]
            def aggregate_layer_weights(weights_matrix):  # type: ignore[no-untyped-def]
                summed = np.sum(weights_matrix, axis=0)
                averaged = summed / weights_matrix.shape[0]
                return averaged.astype(np.int32)

            inputset = []
            for _ in range(8):
                sample = np.random.randint(
                    -self.max_value,
                    self.max_value + 1,
                    size=(self.num_clients, total_elements),
                    dtype=np.int32,
                )
                inputset.append(sample)

            configuration = fhe.Configuration(
                enable_unsafe_features=True,
                use_insecure_key_cache=True,
                insecure_key_cache_location=os.path.join(".keys"),
            )

            try:
                circuit = aggregate_layer_weights.compile(inputset, configuration=configuration)
                circuit.keygen()
                self.circuits[layer_name] = circuit
                if self.verbose:
                    print(f"  ✅ Circuit for {layer_name} compiled successfully")
            except Exception as exc:  # pragma: no cover - unexpected runtime failure
                self.circuits[layer_name] = None
                if self.verbose:
                    print(f"  ❌ Failed to compile circuit for {layer_name}: {exc}")

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------
    def _encrypt_model_weights(self, model_weights_dict: OrderedDict[str, torch.Tensor]) -> Dict[str, np.ndarray]:
        encrypted_weights: Dict[str, np.ndarray] = {}
        for layer_name, weights_tensor in model_weights_dict.items():
            weights_np = self._ensure_cpu_tensor(weights_tensor).numpy()
            scaled_weights = weights_np * self.scale_factor
            clipped = np.clip(scaled_weights, -self.max_value, self.max_value)
            int_weights = np.round(clipped).astype(np.int32)
            encrypted_weights[layer_name] = int_weights.flatten()
        return encrypted_weights

    def aggregate(self, client_weights_list: Iterable[OrderedDict[str, torch.Tensor]]) -> OrderedDict[str, torch.Tensor]:
        encrypted_weights_list = [self._encrypt_model_weights(weights) for weights in client_weights_list]
        aggregated: "OrderedDict[str, torch.Tensor]" = OrderedDict()

        for layer_name in self.layer_order:
            circuit = self.circuits.get(layer_name)
            client_layer_weights = np.array(
                [client_weights[layer_name] for client_weights in encrypted_weights_list],
                dtype=np.int32,
            )

            if circuit is not None:
                try:
                    encrypted_input = circuit.encrypt(client_layer_weights)
                    encrypted_result = circuit.run(encrypted_input)
                    decrypted_result = circuit.decrypt(encrypted_result)
                    float_result = decrypted_result.astype(np.float32) / self.scale_factor
                except Exception as exc:  # pragma: no cover - unexpected runtime failure
                    if self.verbose:
                        print(
                            f"  ⚠️ TFHE aggregation failed for {layer_name}, falling back to plaintext: {exc}"
                        )
                    averaged = np.mean(client_layer_weights, axis=0).astype(np.float32) / self.scale_factor
                    float_result = averaged
            else:
                averaged = np.mean(client_layer_weights, axis=0).astype(np.float32) / self.scale_factor
                float_result = averaged

            reshaped = float_result.reshape(self.weight_shapes[layer_name])
            aggregated[layer_name] = torch.from_numpy(reshaped)

        return aggregated


class CKKSModelAggregator(BaseModelAggregator):
    """Federated model aggregation using OpenFHE's CKKS scheme."""

    def __init__(
        self,
        model_structure: torch.nn.Module,
        num_clients: int,
        scale_factor: float = 1.0,
        multiplicative_depth: int = 3,
        scaling_mod_bits: int = 50,
        verbose: bool = False,
    ) -> None:
        if CCParamsCKKSRNS is None or GenCryptoContext is None:  # pragma: no cover
            raise ImportError(
                "openfhe-python is required for CKKS aggregation but is not installed."
            ) from _OPENFHE_IMPORT_ERROR

        super().__init__(model_structure, num_clients, verbose=verbose)
        self.scale_factor = float(scale_factor)
        self.multiplicative_depth = multiplicative_depth
        self.scaling_mod_bits = scaling_mod_bits
        self.max_slot_count = self._calculate_slot_count()
        self._setup_context()

    def _calculate_slot_count(self) -> int:
        max_elements = max(int(np.prod(shape)) for shape in self.weight_shapes.values())
        # CKKS packing requires power-of-two slots. Ensure at least one slot.
        slots = 1
        while slots < max_elements:
            slots <<= 1
        return slots

    def _setup_context(self) -> None:
        params = CCParamsCKKSRNS()
        params.SetSecurityLevel(SecurityLevel.HEStd_128_classic)
        params.SetMultiplicativeDepth(self.multiplicative_depth)
        params.SetScalingModSize(self.scaling_mod_bits)
        params.SetBatchSize(self.max_slot_count)

        self.context = GenCryptoContext(params)
        self.context.Enable(PKESchemeFeature.PKE)
        self.context.Enable(PKESchemeFeature.KEYSWITCH)
        self.context.Enable(PKESchemeFeature.LEVELEDSHE)

        self.key_pair = self.context.KeyGen()
        self.context.EvalMultKeyGen(self.key_pair.secretKey)
        self.context.EvalSumKeyGen(self.key_pair.secretKey)

    # ------------------------------------------------------------------
    # Aggregation helpers
    # ------------------------------------------------------------------
    def _prepare_plaintext(self, values: np.ndarray):
        padded = np.zeros(self.max_slot_count, dtype=np.float64)
        padded[: values.size] = values.astype(np.float64)
        return self.context.MakeCKKSPackedPlaintext(padded.tolist())

    def _decrypt_to_numpy(self, ciphertext, original_size: int) -> np.ndarray:
        plaintext = self.context.Decrypt(self.key_pair.secretKey, ciphertext)
        decoded: List[float] = plaintext.GetRealPackedValue()
        return np.array(decoded[:original_size], dtype=np.float64)

    def aggregate(self, client_weights_list: Iterable[OrderedDict[str, torch.Tensor]]) -> OrderedDict[str, torch.Tensor]:
        client_weights = list(client_weights_list)
        if not client_weights:
            raise AggregationError("No client weights provided for aggregation.")

        aggregated: "OrderedDict[str, torch.Tensor]" = OrderedDict()

        for layer_name in self.layer_order:
            flat_weights = []
            for weights in client_weights:
                tensor = self._ensure_cpu_tensor(weights[layer_name]).float().numpy() * self.scale_factor
                flat_weights.append(tensor.flatten())

            layer_size = flat_weights[0].size
            cipher_sum = None
            for flat in flat_weights:
                plaintext = self._prepare_plaintext(flat)
                cipher = self.context.Encrypt(self.key_pair.publicKey, plaintext)
                cipher_sum = cipher if cipher_sum is None else self.context.EvalAdd(cipher_sum, cipher)

            avg_cipher = self.context.EvalMult(cipher_sum, 1.0 / self.num_clients)
            decrypted = self._decrypt_to_numpy(avg_cipher, layer_size)
            reshaped = (decrypted / self.scale_factor).reshape(self.weight_shapes[layer_name])
            aggregated[layer_name] = torch.tensor(reshaped, dtype=torch.float32)

        return aggregated


__all__ = [
    "AggregationError",
    "BaseModelAggregator",
    "PlaintextAggregator",
    "TFHEModelAggregator",
    "CKKSModelAggregator",
]

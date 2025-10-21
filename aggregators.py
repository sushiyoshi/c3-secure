"""Aggregation strategies for federated learning experiments."""
from __future__ import annotations

import math
from collections import OrderedDict
from typing import Dict, Iterable, List, Mapping

import numpy as np
import torch

try:  # Optional dependency used only for the TFHE based aggregation
    from concrete import fhe
except ImportError as exc:  # pragma: no cover - handled gracefully at runtime
    fhe = None  # type: ignore

try:  # Optional dependency used only for the CKKS based aggregation
    from openfhe import openfhe as openfhe_lib  # type: ignore
except ImportError:  # pragma: no cover - handled gracefully at runtime
    openfhe_lib = None


StateDict = Mapping[str, torch.Tensor]
AggregatedStateDict = OrderedDict[str, torch.Tensor]


class BaseAggregator:
    """Common functionality for model aggregators."""

    def __init__(self, model_structure: torch.nn.Module, num_clients: int) -> None:
        if num_clients <= 0:
            raise ValueError("num_clients must be a positive integer")

        self.num_clients = num_clients
        self.weight_shapes = OrderedDict(
            (name, param.shape) for name, param in model_structure.named_parameters()
        )

    def aggregate(self, client_weights_list: Iterable[StateDict]) -> AggregatedStateDict:
        raise NotImplementedError


class PlaintextAggregator(BaseAggregator):
    """Performs standard plaintext averaging of model weights."""

    def aggregate(self, client_weights_list: Iterable[StateDict]) -> AggregatedStateDict:
        client_weights = list(client_weights_list)
        if not client_weights:
            raise ValueError("client_weights_list must contain at least one set of weights")

        aggregated: AggregatedStateDict = OrderedDict()
        for layer_name in self.weight_shapes:
            stacked = torch.stack([
                weights[layer_name].detach().cpu() for weights in client_weights
            ])
            aggregated[layer_name] = stacked.mean(dim=0)

        return aggregated


class TFHEModelAggregator(BaseAggregator):
    """Aggregates model weights using Concrete-ML's TFHE backend."""

    def __init__(
        self,
        model_structure: torch.nn.Module,
        num_clients: int,
        scale_factor: int = 100,
        max_value: int = 50,
    ) -> None:
        if fhe is None:
            raise ImportError(
                "concrete-ml is required for TFHE aggregation but is not installed."
            )

        super().__init__(model_structure, num_clients)
        self.scale_factor = scale_factor
        self.max_value = max_value
        self.circuits: Dict[str, "fhe.Circuit"] = {}
        self._compile_fhe_circuits()

    def _compile_fhe_circuits(self) -> None:
        """Compile TFHE circuits for each model layer."""
        for layer_name, shape in self.weight_shapes.items():
            total_elements = int(np.prod(shape))

            @fhe.compiler({"weights_matrix": "encrypted"})
            def aggregate_layer_weights(weights_matrix: np.ndarray) -> np.ndarray:
                summed = np.sum(weights_matrix, axis=0)
                averaged = summed / weights_matrix.shape[0]
                return averaged.astype(np.int32)

            inputset: List[np.ndarray] = []
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
                insecure_key_cache_location=".keys",
            )

            try:
                circuit = aggregate_layer_weights.compile(inputset, configuration=configuration)
                circuit.keygen()
                self.circuits[layer_name] = circuit
            except Exception as exc:  # pragma: no cover - compilation failures are rare
                self.circuits[layer_name] = None
                print(f"[TFHE] Failed to compile circuit for {layer_name}: {exc}")

    def _encrypt_model_weights(self, model_weights_dict: StateDict) -> Dict[str, np.ndarray]:
        encrypted_weights: Dict[str, np.ndarray] = {}
        for layer_name, weights_tensor in model_weights_dict.items():
            weights_np = weights_tensor.detach().cpu().numpy()
            scaled_weights = weights_np * self.scale_factor
            clipped_weights = np.clip(scaled_weights, -self.max_value, self.max_value)
            int_weights = np.round(clipped_weights).astype(np.int32)
            encrypted_weights[layer_name] = int_weights.flatten()
        return encrypted_weights

    def aggregate(self, client_weights_list: Iterable[StateDict]) -> AggregatedStateDict:
        client_weights = list(client_weights_list)
        if not client_weights:
            raise ValueError("client_weights_list must contain at least one set of weights")

        encrypted_weights_list = [
            self._encrypt_model_weights(client_weights_dict)
            for client_weights_dict in client_weights
        ]

        aggregated: AggregatedStateDict = OrderedDict()
        for layer_name, shape in self.weight_shapes.items():
            circuit = self.circuits.get(layer_name)
            client_layer_weights = np.stack(
                [client_weights[layer_name] for client_weights in encrypted_weights_list],
                axis=0,
            )

            if circuit is not None:
                try:
                    encrypted_input = circuit.encrypt(client_layer_weights)
                    encrypted_result = circuit.run(encrypted_input)
                    decrypted_result = circuit.decrypt(encrypted_result)
                    float_result = decrypted_result.astype(np.float32) / self.scale_factor
                except Exception as exc:  # pragma: no cover - runtime failures are rare
                    print(f"[TFHE] Runtime failure for {layer_name}, falling back to plaintext: {exc}")
                    averaged = np.mean(client_layer_weights, axis=0)
                    float_result = averaged.astype(np.float32) / self.scale_factor
            else:
                averaged = np.mean(client_layer_weights, axis=0)
                float_result = averaged.astype(np.float32) / self.scale_factor

            reshaped = float_result.reshape(shape)
            aggregated[layer_name] = torch.from_numpy(reshaped)

        return aggregated


class CKKSModelAggregator(BaseAggregator):
    """Aggregates model weights using OpenFHE's CKKS scheme."""

    def __init__(
        self,
        model_structure: torch.nn.Module,
        num_clients: int,
        mult_depth: int = 2,
        scale_mod_size: int = 40,
    ) -> None:
        if openfhe_lib is None:
            raise ImportError(
                "openfhe-python is required for CKKS aggregation but is not installed."
            )

        super().__init__(model_structure, num_clients)
        self.mult_depth = mult_depth
        self.scale_mod_size = scale_mod_size

        max_elements = max(int(np.prod(shape)) for shape in self.weight_shapes.values())
        self.batch_size = self._next_power_of_two(max_elements)

        self.crypto_context = self._create_context()
        self.key_pair = self.crypto_context.KeyGen()
        self.crypto_context.EvalSumKeyGen(self.key_pair.secretKey)
        self.crypto_context.EvalMultKeyGen(self.key_pair.secretKey)

    @staticmethod
    def _next_power_of_two(value: int) -> int:
        return 1 if value <= 1 else int(2 ** math.ceil(math.log2(value)))

    def _create_context(self):
        params = openfhe_lib.CCParamsCKKSRNS()
        params.SetMultiplicativeDepth(self.mult_depth)
        params.SetScalingModSize(self.scale_mod_size)
        params.SetBatchSize(self.batch_size)
        params.SetSecurityLevel(openfhe_lib.SecurityLevel.HEStd_128_classic)

        cc = openfhe_lib.GenCryptoContext(params)
        cc.Enable(openfhe_lib.PKESchemeFeature.PKE)
        cc.Enable(openfhe_lib.PKESchemeFeature.KEYSWITCH)
        cc.Enable(openfhe_lib.PKESchemeFeature.LEVELEDSHE)
        cc.Enable(openfhe_lib.PKESchemeFeature.ADVANCEDSHE)
        return cc

    def _encrypt_tensor(self, tensor: torch.Tensor):
        flat = tensor.detach().cpu().numpy().astype(np.float64).reshape(-1)
        padded = np.zeros(self.batch_size, dtype=np.float64)
        padded[: flat.size] = flat
        plaintext = self.crypto_context.MakeCKKSPackedPlaintext(padded.tolist())
        return self.crypto_context.Encrypt(self.key_pair.publicKey, plaintext)

    def _decrypt_ciphertext(self, ciphertext, original_size: int) -> np.ndarray:
        plaintext = self.crypto_context.Decrypt(self.key_pair.secretKey, ciphertext)
        plaintext.SetLength(self.batch_size)
        values = np.array(plaintext.GetRealPackedValue(), dtype=np.float64)
        return values[:original_size]

    def aggregate(self, client_weights_list: Iterable[StateDict]) -> AggregatedStateDict:
        client_weights = list(client_weights_list)
        if not client_weights:
            raise ValueError("client_weights_list must contain at least one set of weights")

        aggregated: AggregatedStateDict = OrderedDict()
        num_clients = len(client_weights)

        for layer_name, shape in self.weight_shapes.items():
            ciphertext_sum = None
            for weights in client_weights:
                ciphertext = self._encrypt_tensor(weights[layer_name])
                if ciphertext_sum is None:
                    ciphertext_sum = ciphertext
                else:
                    ciphertext_sum = self.crypto_context.EvalAdd(ciphertext_sum, ciphertext)

            if ciphertext_sum is None:
                raise RuntimeError("No ciphertexts produced during CKKS aggregation")

            scale_factor = 1.0 / num_clients
            ciphertext_avg = self.crypto_context.EvalMult(ciphertext_sum, scale_factor)
            decrypted = self._decrypt_ciphertext(ciphertext_avg, int(np.prod(shape)))
            reshaped = decrypted.reshape(shape)
            aggregated[layer_name] = torch.from_numpy(reshaped.astype(np.float32))

        return aggregated


def clone_state_dict(state_dict: StateDict) -> AggregatedStateDict:
    """Deep copy a PyTorch state dict to detach it from the computation graph."""
    cloned: AggregatedStateDict = OrderedDict()
    for name, tensor in state_dict.items():
        cloned[name] = tensor.detach().cpu().clone()
    return cloned

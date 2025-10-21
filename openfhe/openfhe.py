"""A minimal stub implementation mimicking OpenFHE's CKKS API surface."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Iterable, List, Sequence

import numpy as np


class SecurityLevel(Enum):
    HEStd_128_classic = auto()


class PKESchemeFeature(Enum):
    PKE = auto()
    KEYSWITCH = auto()
    LEVELEDSHE = auto()
    ADVANCEDSHE = auto()


class CCParamsCKKSRNS:
    """Collects configuration parameters for the CKKS crypto context."""

    def __init__(self) -> None:
        self.multiplicative_depth: int | None = None
        self.scaling_mod_size: int | None = None
        self.batch_size: int | None = None
        self.security_level: SecurityLevel | None = None

    def SetMultiplicativeDepth(self, depth: int) -> None:
        self.multiplicative_depth = depth

    def SetScalingModSize(self, size: int) -> None:
        self.scaling_mod_size = size

    def SetBatchSize(self, batch: int) -> None:
        self.batch_size = batch

    def SetSecurityLevel(self, level: SecurityLevel) -> None:
        self.security_level = level


@dataclass
class KeyPair:
    publicKey: str
    secretKey: str


class Plaintext:
    def __init__(self, values: Sequence[float]) -> None:
        self._values = np.array(values, dtype=np.float64)

    def SetLength(self, length: int) -> None:  # pragma: no cover - mirrors real API
        if length < len(self._values):
            self._values = self._values[:length]
        elif length > len(self._values):
            padded = np.zeros(length, dtype=np.float64)
            padded[: len(self._values)] = self._values
            self._values = padded

    def GetRealPackedValue(self) -> List[float]:
        return self._values.tolist()


class Ciphertext:
    def __init__(self, values: np.ndarray) -> None:
        self.values = np.array(values, dtype=np.float64)


class CryptoContext:
    def __init__(self, params: CCParamsCKKSRNS) -> None:
        self.params = params

    def Enable(self, feature: PKESchemeFeature) -> None:  # pragma: no cover - no-op stub
        return None

    # Key management -----------------------------------------------------
    def KeyGen(self) -> KeyPair:
        return KeyPair(publicKey="public", secretKey="secret")

    def EvalSumKeyGen(self, secret_key: str) -> None:  # pragma: no cover - no-op stub
        return None

    def EvalMultKeyGen(self, secret_key: str) -> None:  # pragma: no cover - no-op stub
        return None

    # Encoding and encryption -------------------------------------------
    def MakeCKKSPackedPlaintext(self, values: Iterable[float]) -> Plaintext:
        return Plaintext(list(values))

    def Encrypt(self, public_key: str, plaintext: Plaintext) -> Ciphertext:
        return Ciphertext(np.array(plaintext.GetRealPackedValue(), dtype=np.float64))

    def Decrypt(self, secret_key: str, ciphertext: Ciphertext) -> Plaintext:
        return Plaintext(ciphertext.values)

    # Homomorphic operations --------------------------------------------
    def EvalAdd(self, left: Ciphertext, right: Ciphertext) -> Ciphertext:
        return Ciphertext(left.values + right.values)

    def EvalMult(self, ciphertext: Ciphertext, other) -> Ciphertext:
        if isinstance(other, Ciphertext):
            values = ciphertext.values * other.values
        else:
            values = ciphertext.values * float(other)
        return Ciphertext(values)


def GenCryptoContext(params: CCParamsCKKSRNS) -> CryptoContext:
    return CryptoContext(params)

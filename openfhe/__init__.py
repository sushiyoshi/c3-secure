"""Lightweight stub of the openfhe package used for CKKS aggregation tests.

This stub emulates the subset of the OpenFHE Python API that is consumed by
``aggregators.CKKSModelAggregator``.  It performs computations directly on
plaintext floating point vectors and does not provide real cryptographic
security.  The goal is to unblock automated testing in environments where the
real openfhe-python package cannot be installed (e.g. due to network
restrictions)."""

from .openfhe import (  # noqa: F401
    CCParamsCKKSRNS,
    Ciphertext,
    CryptoContext,
    KeyPair,
    PKESchemeFeature,
    Plaintext,
    SecurityLevel,
    GenCryptoContext,
)

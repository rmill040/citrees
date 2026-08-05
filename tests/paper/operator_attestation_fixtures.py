"""Deterministic Ed25519 key material for paper-pipeline tests."""

from __future__ import annotations

from Crypto.PublicKey import ECC

from paper.benchmark.pipeline.operator_attestation import (
    operator_public_key_from_private_key,
)

OPERATOR_PRIVATE_KEY_PEM = ECC.construct(
    curve="Ed25519",
    seed=bytes(range(32)),
).export_key(format="PEM").encode("ascii")
OPERATOR_PUBLIC_KEY = operator_public_key_from_private_key(OPERATOR_PRIVATE_KEY_PEM)

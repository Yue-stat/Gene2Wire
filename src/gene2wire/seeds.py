"""Stable seed derivation independent of Python hash randomization."""

from __future__ import annotations

import hashlib
import json
from typing import Any


def stable_seed(base_seed: int, *parts: Any) -> int:
    """Derive a deterministic NumPy-compatible seed from semantic parts."""

    payload = json.dumps(
        {"base_seed": int(base_seed), "parts": parts},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**32 - 1)


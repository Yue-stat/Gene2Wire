"""Atomic local caching; successful raw downloads are reusable offline."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
from urllib.request import Request, urlopen

import numpy as np


def file_hash(path: str | Path, *, git_blob: bool = False) -> str:
    path = Path(path)
    digest = hashlib.sha1() if git_blob else hashlib.sha256()
    if git_blob:
        digest.update(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def jsonable(value):
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def atomic_json(value, destination: str | Path) -> None:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=destination.name + ".", suffix=".tmp",
                                      dir=destination.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(jsonable(value), handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def cached_download(url: str, destination: str | Path, *, sha256: str | None = None,
                    git_blob_sha: str | None = None, timeout: int = 120) -> Path:
    """Reuse a nonempty validated file without contacting the server.

    If the upstream publisher provides no checksum, a local SHA256 records the
    first accepted bytes; it is an integrity check, not proof of provenance.
    Interrupted downloads never replace the last complete file.
    """
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    manifest = destination.with_name(destination.name + ".download.json")

    def verify(path: Path) -> str:
        if path.stat().st_size == 0:
            raise ValueError(f"Cached/downloaded file is empty: {path}")
        digest = file_hash(path)
        if sha256 is not None and digest != sha256:
            raise ValueError(f"SHA256 mismatch for {path.name}")
        if git_blob_sha is not None and file_hash(path, git_blob=True) != git_blob_sha:
            raise ValueError(f"Git blob checksum mismatch for {path.name}")
        return digest

    if destination.exists():
        digest = verify(destination)
        if manifest.exists():
            recorded = json.loads(manifest.read_text())
            if recorded.get("sha256") != digest:
                raise ValueError(f"Raw cache changed: {destination}; preserve or remove it explicitly")
        else:
            atomic_json({"url": url, "sha256": digest, "bytes": destination.stat().st_size,
                         "publisher_checksum": sha256 or git_blob_sha,
                         "adopted_existing_file": True}, manifest)
        return destination

    fd, temporary_name = tempfile.mkstemp(prefix=destination.name + ".", suffix=".part",
                                         dir=destination.parent)
    temporary = Path(temporary_name)
    try:
        request = Request(url, headers={"User-Agent": "Gene2Wire/0908 reproducible research"})
        with os.fdopen(fd, "wb") as handle, urlopen(request, timeout=timeout) as response:
            for chunk in iter(lambda: response.read(1024 * 1024), b""):
                handle.write(chunk)
            handle.flush()
            os.fsync(handle.fileno())
        digest = verify(temporary)
        os.replace(temporary, destination)
        atomic_json({"url": url, "sha256": digest, "bytes": destination.stat().st_size,
                     "publisher_checksum": sha256 or git_blob_sha,
                     "adopted_existing_file": False}, manifest)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def atomic_npz(destination: str | Path, **arrays) -> None:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=destination.name + ".", suffix=".npz",
                                      dir=destination.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)

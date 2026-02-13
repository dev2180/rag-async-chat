"""
MODULE: app/ingestion/hasher.py

Computes SHA256 hash of file content.
Used for document identity.
"""

import hashlib
from pathlib import Path


def compute_file_hash(file_path: Path) -> str:
    sha256 = hashlib.sha256()

    with file_path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)

    return sha256.hexdigest()

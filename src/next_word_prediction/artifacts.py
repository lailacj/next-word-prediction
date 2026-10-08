"""Small helpers for traceable files and atomic publication."""

import hashlib
import json
import os
from pathlib import Path
import tempfile


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as file:
        for block in iter(lambda: file.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def publish(source, destination, *, overwrite):
    """Publish a staged file, refusing a destination created concurrently."""
    if overwrite:
        os.replace(source, destination)
    else:
        os.link(source, destination)
        Path(source).unlink()


def write_json(path, value):
    path = Path(path)
    with tempfile.NamedTemporaryFile('w', encoding='utf-8', dir=path.parent, delete=False) as file:
        temporary = Path(file.name)
        try:
            json.dump(value, file, indent=2, allow_nan=False)
            file.write('\n')
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)

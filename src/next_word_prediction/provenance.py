"""Record enough context to identify inputs, code, and the run environment."""

from datetime import datetime, timezone
from importlib.metadata import distributions
from pathlib import Path
import platform
import subprocess


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def environment():
    packages = {d.metadata['Name']: d.version for d in distributions() if d.metadata['Name']}
    return {"python": platform.python_version(), "platform": platform.platform(),
            "packages": dict(sorted(packages.items()))}


def code_version():
    root = Path(__file__).resolve().parents[2]
    result = {"git_commit": None, "git_dirty": None, "source_sha256": {}}
    from .artifacts import sha256
    result['source_sha256'] = {p.name: sha256(p) for p in Path(__file__).parent.glob('*.py')}
    if (root / '.git').exists():
        try:
            result['git_commit'] = subprocess.check_output(
                ['git', 'rev-parse', 'HEAD'], cwd=root, stderr=subprocess.DEVNULL, text=True,
            ).strip()
            result['git_dirty'] = bool(subprocess.check_output(
                ['git', 'status', '--porcelain'], cwd=root, stderr=subprocess.DEVNULL, text=True,
            ).strip())
        except (OSError, subprocess.CalledProcessError):
            pass
    return result

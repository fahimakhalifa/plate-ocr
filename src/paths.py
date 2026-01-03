from __future__ import annotations

from pathlib import Path
from typing import Iterable, Union


def repo_root() -> Path:
    # <repo>/src/paths.py -> parents[1] = <repo>
    return Path(__file__).resolve().parents[1]


def _candidates(relative: Union[str, Path]) -> Iterable[Path]:
    rel = Path(relative)

    # 1) current working directory (preserve current behavior)
    yield (Path.cwd() / rel).resolve()

    # 2) repo root
    yield (repo_root() / rel).resolve()


def resolve_path(relative: Union[str, Path]) -> str:
    """
    Resolve a relative path robustly.
    - First checks CWD-relative (preserves existing behavior)
    - Then repo-root-relative
    - If not found, returns repo-root-relative (for clear error messages downstream)
    """
    last = None
    for p in _candidates(relative):
        last = p
        if p.exists():
            return str(p)
    return str(last) if last else str((repo_root() / Path(relative)).resolve())


# ---- Model asset helpers (new canonical locations) ----

def model_asset(filename: str) -> str:
    """
    Canonical location: assets/models/<filename>
    Backward compatible with:
    - <repo>/<filename> (old LP-detection.pt at root)
    - model/<filename>  (old CRNN assets)
    """
    # New canonical path
    new_path = Path("assets") / "models" / filename
    p_new = Path(resolve_path(new_path))
    if p_new.exists():
        return str(p_new)

    # Back-compat 1: root file
    p_root = Path(resolve_path(filename))
    if p_root.exists():
        return str(p_root)

    # Back-compat 2: old model folder
    p_old = Path(resolve_path(Path("model") / filename))
    return str(p_old)

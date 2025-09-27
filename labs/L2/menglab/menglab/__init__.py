# menglab/__init__.py

from .load import (
    load_dataset,
    load_full_dataset,
    load_alkanes,
    load_expanded_alkanes,
    load_multilinear_alkanes,
)

__all__ = [
    "load_dataset",
    "load_full_dataset",
    "load_alkanes",
    "load_expanded_alkanes",
    "load_multilinear_alkanes",
]

# Optional: add version metadata
try:
    from importlib.metadata import version as _pkg_version
    __version__ = _pkg_version("menglab")
except Exception:
    __version__ = "0.0.0"

# menglab/__init__.py

from .hello import (
    hello
)

__all__ = [
    # hello
    "hello"
]

# Optional: add version metadata
try:
    from importlib.metadata import version as _pkg_version
    __version__ = _pkg_version("menglab")
except Exception:
    __version__ = "0.0.0"

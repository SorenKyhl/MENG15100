# menglab/__init__.py

from .load import (
)

from .gui import (
    interactive_activation_functions
)


__all__ = [
    # load
    "interactive_activation_functions"
    # gui
]

# Optional: add version metadata
try:
    from importlib.metadata import version as _pkg_version
    __version__ = _pkg_version("menglab")
except Exception:
    __version__ = "0.0.0"

# menglab/__init__.py

from .load import (
)

from .gui import (
    interactive_activation_functions,
    interactive_single_layer_model
)


__all__ = [
    # load
    # gui
    "interactive_activation_functions",
    "interactive_single_layer_model"
]

# Optional: add version metadata
try:
    from importlib.metadata import version as _pkg_version
    __version__ = _pkg_version("menglab")
except Exception:
    __version__ = "0.0.0"

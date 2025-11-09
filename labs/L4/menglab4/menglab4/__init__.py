# menglab/__init__.py

from .load import (
    generate_circle_data
)

from .gui import (
    interactive_activation_functions,
    interactive_single_layer_model
)

from .pytorch_training import(
    set_pytorch_seed,
    plot_model_predictions_pytorch,
    plot_pytorch_training
    plot_circle_results


)

__all__ = [
    # load
    "generate_circle_data",
    # gui
    "interactive_activation_functions",
    "interactive_single_layer_model",
    # pytorch_training
    "set_pytorch_seed",
    "plot_model_predictions_pytorch",
    "plot_pytorch_training",
    "plot_circle_results"
]

# Optional: add version metadata
try:
    from importlib.metadata import version as _pkg_version
    __version__ = _pkg_version("menglab")
except Exception:
    __version__ = "0.0.0"

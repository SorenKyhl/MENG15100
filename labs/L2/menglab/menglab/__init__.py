# menglab/__init__.py

from .load import (
    load_dataset,
    load_full_dataset,
    load_alkanes,
    load_expanded_alkanes,
    load_multilinear_alkanes,
)

from .gui import (
    data_exploere,
    interactive_loss_landscape,
    manually_train_linear_regression,
    interactive_gradient_descent
)

__all__ = [
    # load
    "load_dataset",
    "load_full_dataset",
    "load_alkanes",
    "load_expanded_alkanes",
    "load_multilinear_alkanes",
    # gui
    "data_explorer",
    "interactive_loss_landscape",
    "manually_train_linear_regression",
    "interactive_gradient_descent",
]

# Optional: add version metadata
try:
    from importlib.metadata import version as _pkg_version
    __version__ = _pkg_version("menglab")
except Exception:
    __version__ = "0.0.0"

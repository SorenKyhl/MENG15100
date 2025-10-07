# menglab/__init__.py

from .load import (
    generate_projection_data,
    generate_PCA_data,
    generate_kmeans_data
)

__all__ = [
    # load
    "generate_projection_data", 
    "generate_PCA_data",
    "generate_kmeans_data"
]

# Optional: add version metadata
try:
    from importlib.metadata import version as _pkg_version
    __version__ = _pkg_version("menglab")
except Exception:
    __version__ = "0.0.0"

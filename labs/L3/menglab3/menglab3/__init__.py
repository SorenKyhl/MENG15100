# menglab/__init__.py

from .load import (
    generate_projection_data,
    generate_PCA_data,
    generate_kmeans_data
)

from .gui import (
    interactive_projection,
    interactive_PCA_rotation,
    interactive_PCA_projection,
    scree_plot,
    pca_scatter,
    elbow_plot,
    silhouette_scan,
    interactive_2d_projection,
    PCA_projection_3D,
    interactive_k_means
)


__all__ = [
    # load
    "generate_projection_data", 
    "generate_PCA_data",
    "generate_kmeans_data",
    # gui
    "interactive_projection", 
    "interactive_PCA_rotation",
    "interactive_PCA_projection",
    "scree_plot",
    "pca_scatter", 
    "elbow_plot",
    "silhouette_scan",
    "interactive_2d_projection",
    "PCA_projection_3D",
    "interactive_k_means"
]

# Optional: add version metadata
try:
    from importlib.metadata import version as _pkg_version
    __version__ = _pkg_version("menglab")
except Exception:
    __version__ = "0.0.0"

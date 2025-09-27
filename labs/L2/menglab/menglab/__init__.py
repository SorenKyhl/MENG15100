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

try:
    from rdkit import Chem  # and other rdkit imports as needed
except Exception as e:
    raise ImportError(
        "menglab requires RDKit. Install it via conda-forge, e.g.:\n"
        "  conda install -c conda-forge rdkit\n"
        "or create a new env with rdkit preinstalled."
    ) from e

try:
    from rdkit import Chem  # and other rdkit imports as needed
except Exception as e:
    raise ImportError(
        "menglab requires RDKit. Install it via conda-forge, e.g.:\n"
        "  conda install -c conda-forge rdkit\n"
        "or create a new env with rdkit preinstalled."
    ) from e

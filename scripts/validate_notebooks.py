from pathlib import Path
import nbformat
for path in Path("notebooks").glob("*.ipynb"):
    nbformat.validate(nbformat.read(path, as_version=4))
    print("Valid:", path)

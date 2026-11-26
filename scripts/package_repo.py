"""Package only source files, never data, model weights, tokens, or Git metadata."""
from pathlib import Path
import zipfile
ROOT = Path(__file__).resolve().parents[1]
archive = ROOT.parent / "llama3-qlora-finance-colab.zip"
allowed_dirs = {"finance_qlora", "notebooks", "configs", "docs", "tests", "scripts", ".github"}
allowed_files = {"README.md", "LICENSE", "pyproject.toml", "requirements-cpu.txt", "requirements-colab.txt", ".gitignore"}
with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as z:
    for p in sorted(ROOT.rglob("*")):
        relative = p.relative_to(ROOT)
        if not p.is_file() or "__pycache__" in relative.parts:
            continue
        if relative.parts[0] in allowed_dirs or str(relative) in allowed_files:
            z.write(p, arcname=str(Path("llama3-qlora-finance-colab")/relative))
print(archive)

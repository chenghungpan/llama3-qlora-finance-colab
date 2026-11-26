import ast
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
class ArtifactTests(unittest.TestCase):
    def test_notebooks_have_valid_python_and_no_saved_outputs(self):
        notebooks = list((ROOT/"notebooks").glob("*.ipynb"))
        self.assertEqual(len(notebooks), 3)
        for path in notebooks:
            notebook = json.loads(path.read_text())
            self.assertEqual(notebook["nbformat"], 4)
            for cell in notebook["cells"]:
                if cell["cell_type"] == "code":
                    ast.parse("".join(cell["source"]))
                    self.assertEqual(cell["outputs"], [])
                    self.assertIsNone(cell["execution_count"])
    def test_configs_have_consistent_keys(self):
        configs = [json.loads(p.read_text()) for p in (ROOT/"configs").glob("*.json")]
        self.assertEqual(len(configs), 3)
        self.assertTrue(all(c.keys() == configs[0].keys() for c in configs))

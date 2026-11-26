"""Optional CPU integration: no model download, credentials, or GPU."""
import importlib.util
import tempfile
import unittest
from pathlib import Path

HAS_SKLEARN = importlib.util.find_spec("sklearn") is not None
HAS_TORCH = importlib.util.find_spec("torch") is not None

class OptionalMLTests(unittest.TestCase):
    @unittest.skipUnless(HAS_SKLEARN, "Install requirements-cpu.txt")
    def test_cpu_baselines(self):
        from finance_qlora.baselines import evaluate_baselines
        rows = [{"text": text, "label": label} for label, text in
                [(0,"profits fall losses"),(0,"losses grow"),(1,"profits rise gains"),
                 (1,"gains grow"),(2,"meeting scheduled"),(2,"meeting announced")]]
        result = evaluate_baselines(rows, [{"text":"profits rise", "label":1}, {"text":"losses", "label":0}])
        self.assertEqual(result["tfidf_logistic"]["n"], 2)
        self.assertEqual(result["tfidf_logistic"]["invalid_rate"], 0)

    @unittest.skipUnless(HAS_TORCH, "Install GPU requirements or CPU torch")
    def test_collator_preserves_real_eos(self):
        from finance_qlora.modeling import CompletionCollator
        rows = [{"input_ids":[1,9,2],"attention_mask":[1,1,1],"labels":[-100,9,2]},
                {"input_ids":[1,2],"attention_mask":[1,1],"labels":[-100,2]}]
        batch = CompletionCollator(2)(rows)
        self.assertEqual(batch["labels"].tolist(), [[-100,9,2],[-100,2,-100]])
        self.assertEqual(batch["attention_mask"].tolist(), [[1,1,1],[1,1,0]])

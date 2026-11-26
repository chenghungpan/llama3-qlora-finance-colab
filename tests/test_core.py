import unittest
from finance_qlora.core import clean_splits, encode_example, example_id, metrics, parse_label, subset

class TinyTokenizer:
    bos_token_id = 1
    eos_token_id = 2
    def encode(self, text, add_special_tokens=False):
        return [ord(c)+3 for c in text]

class CoreTests(unittest.TestCase):
    def rows(self):
        return [{"text": f"class {label} unique {i}", "label": label} for label in range(3) for i in range(8)]

    def test_mapping_and_strict_format(self):
        self.assertEqual(parse_label(" Bullish\n"), "bullish")
        self.assertEqual(parse_label("bullish because profits rose"), "invalid")
        self.assertEqual(parse_label("not bearish"), "invalid")

    def test_duplicates_conflicts_and_holdout(self):
        train = self.rows() + [{"text": " SAME   NEWS ", "label": 0},
            {"text": "conflict", "label": 1}, {"text": "Conflict", "label": 2}]
        test = [{"text": "same news", "label": 0}, {"text": "test-only", "label": 2}]
        splits, audit = clean_splits(train, test)
        self.assertEqual(audit["conflicting_groups_removed"], 1)
        self.assertEqual(audit["cross_split_groups"], 1)
        self.assertIn(example_id("same news"), {r["id"] for r in splits["test"]})
        sets = [set(r["id"] for r in rows) for rows in splits.values()]
        self.assertFalse(sets[0]&sets[1] or sets[0]&sets[2] or sets[1]&sets[2])
        self.assertEqual(splits, clean_splits(train, test)[0])

    def test_all_classes_in_validation(self):
        splits, _ = clean_splits(self.rows(), [{"text": "held out", "label": 0}])
        self.assertEqual({r["label"] for r in splits["validation"]}, {0, 1, 2})
        with self.assertRaises(ValueError):
            clean_splits([{"text": "bad", "label": 9}], [])

    def test_subsets_are_reproducible_and_balanced(self):
        a = subset(self.rows(), 9)
        self.assertEqual(a, subset(self.rows(), 9))
        self.assertEqual([sum(r["label"] == k for r in a) for k in range(3)], [3, 3, 3])

    def test_prompt_loss_mask_and_eos_survive_truncation(self):
        tokenizer = TinyTokenizer()
        row = encode_example(tokenizer, {"text": "long "*300, "label": 1}, 256)
        target = tokenizer.encode("bullish") + [2]
        self.assertLessEqual(len(row["input_ids"]), 256)
        self.assertEqual([x for x in row["labels"] if x != -100], target)
        boundary = len(row["input_ids"]) - len(target)
        self.assertTrue(all(x == -100 for x in row["labels"][:boundary]))
        cue = tokenizer.encode("\nSentiment:\n")
        self.assertEqual(row["input_ids"][boundary-len(cue):boundary], cue)
        self.assertEqual(row["input_ids"][-1], 2)

    def test_invalid_outputs_count_as_errors(self):
        m = metrics(["bearish", "bullish", "neutral"], ["bearish", "invalid", "neutral"])
        self.assertAlmostEqual(m["accuracy"], 2/3)
        self.assertAlmostEqual(m["macro_f1"], 2/3)
        self.assertEqual(m["confusion_matrix"][1][3], 1)
        self.assertAlmostEqual(m["invalid_rate"], 1/3)

if __name__ == "__main__": unittest.main()

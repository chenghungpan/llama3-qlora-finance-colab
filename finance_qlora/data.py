"""Download pinned CSV files without executing dataset-provided code."""
import argparse
import csv
import hashlib
import json
import shutil
import urllib.request
from collections import Counter
from pathlib import Path
from .core import DATASET_ID, DATASET_REVISION, clean_splits, save_json, write_jsonl

EXPECTED_SHA256 = {
    "sent_train.csv": "71aa99d0d37feced03c999ec23b5b25302c92f39d69b6acb0a37aa076c5420e8",
    "sent_valid.csv": "31ea8274c98b52212daa68f2fb962811bf1913c1efe51b99765c8e2311806de6",
}

def prepare(output="data", source_dir=None):
    output = Path(output)
    raw = output / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    rows, checksums = {}, {}
    for split, filename in (("train", "sent_train.csv"), ("test", "sent_valid.csv")):
        path = raw / filename
        if source_dir:
            shutil.copyfile(Path(source_dir)/filename, path)
        else:
            url = f"https://huggingface.co/datasets/{DATASET_ID}/resolve/{DATASET_REVISION}/{filename}"
            with urllib.request.urlopen(url, timeout=120) as response:
                path.write_bytes(response.read())
        checksums[filename] = hashlib.sha256(path.read_bytes()).hexdigest()
        if not source_dir and checksums[filename] != EXPECTED_SHA256[filename]:
            raise ValueError(f"Unexpected checksum for {filename}; refusing changed data")
        with path.open(encoding="utf-8-sig", newline="") as f:
            rows[split] = list(csv.DictReader(f))
    splits, audit = clean_splits(rows["train"], rows["test"])
    for name, split in splits.items():
        write_jsonl(output/f"{name}.jsonl", split)
    manifest = {"dataset": DATASET_ID, "revision": DATASET_REVISION,
        "source": "local CSV override" if source_dir else "pinned Hugging Face download",
        "seed": 42, "upstream_validation_used_as": "test", "sha256": checksums,
        "audit": audit, "counts": {k: len(v) for k, v in splits.items()},
        "labels": {k: dict(Counter(str(r["label"]) for r in v)) for k, v in splits.items()},
        "limitations": ["Exact normalized duplicate removal only; semantic duplicates may remain.",
                         "Random classification split; not suitable for a forecasting backtest."]}
    save_json(output/"manifest.json", manifest)
    print(json.dumps(manifest, indent=2))
    return manifest

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="data")
    p.add_argument("--source-dir", help="Offline override containing the two original CSVs")
    a = p.parse_args(); prepare(a.output, a.source_dir)

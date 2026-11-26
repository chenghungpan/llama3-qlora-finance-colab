"""Reload only the saved adapter plus its pinned base model."""
import argparse
import json
from pathlib import Path
from .core import MODEL_ID, MODEL_REVISION, PROMPT_VERSION

def predict(adapter, text):
    from peft import PeftModel
    from .modeling import generate_label, load_quantized
    identity = json.loads((Path(adapter)/"tutorial_config.json").read_text())
    if (identity["model"], identity["model_revision"], identity["prompt_version"]) != (MODEL_ID, MODEL_REVISION, PROMPT_VERSION):
        raise ValueError("Adapter was produced with a different base model or prompt")
    model, tokenizer, _ = load_quantized()
    model = PeftModel.from_pretrained(model, adapter, is_trainable=False)
    return generate_label(model, tokenizer, text, identity["config"]["max_length"])

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--adapter", required=True)
    p.add_argument("--text", required=True)
    a = p.parse_args(); print(json.dumps(predict(a.adapter, a.text), indent=2))

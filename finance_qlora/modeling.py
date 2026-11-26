"""Lazy GPU dependencies keep data lessons and tests CPU-only."""
import json
import time
from pathlib import Path
from .core import MODEL_ID, MODEL_REVISION, PROMPT_VERSION, prompt_ids, parse_label

def load_quantized():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    if not torch.cuda.is_available():
        raise RuntimeError("QLoRA requires an NVIDIA GPU. In Colab select Runtime > Change runtime type > GPU.")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("This tutorial supports a single visible GPU. Set CUDA_VISIBLE_DEVICES=0.")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, revision=MODEL_REVISION,
        quantization_config=config, torch_dtype=dtype, device_map={"": 0},
        attn_implementation="sdpa")
    model.config.pad_token_id = tokenizer.pad_token_id
    return model, tokenizer, dtype

class CompletionCollator:
    """Right-pad inputs; mask padding separately even when PAD and EOS share an ID."""
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, rows):
        import torch
        length = max(len(r["input_ids"]) for r in rows)
        batch = {key: [] for key in ("input_ids", "attention_mask", "labels")}
        for row in rows:
            n = length - len(row["input_ids"])
            batch["input_ids"].append(row["input_ids"] + [self.pad_token_id]*n)
            batch["attention_mask"].append(row["attention_mask"] + [0]*n)
            batch["labels"].append(row["labels"] + [-100]*n)
        return {k: torch.tensor(v, dtype=torch.long) for k, v in batch.items()}

def generate_label(model, tokenizer, text, max_length=256):
    import torch
    source = prompt_ids(tokenizer, text, max_length)
    inputs = torch.tensor([source], device=model.device)
    model.eval()
    with torch.inference_mode():
        output = model.generate(input_ids=inputs, attention_mask=torch.ones_like(inputs),
            max_new_tokens=8, do_sample=False, pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id, use_cache=True)
    answer = output[0, len(source):]
    raw = tokenizer.decode(answer, skip_special_tokens=True)
    return {"prediction": parse_label(raw), "generated_text": raw,
            "input_tokens": len(source), "output_tokens": len(answer)}

def evaluate_model(model, tokenizer, rows, output, max_length=256):
    from .core import LABELS, metrics, write_jsonl, save_json
    predictions, telemetry = [], []
    started = time.perf_counter()
    for i, row in enumerate(rows):
        t0 = time.perf_counter()
        answer = generate_label(model, tokenizer, row["text"], max_length)
        predictions.append({"id": row["id"], "truth": LABELS[row["label"]], **answer})
        telemetry.append({"provider": "local_huggingface", "model": MODEL_ID,
            "purpose": "sentiment_evaluation", "prompt_version": PROMPT_VERSION,
            "latency_seconds": time.perf_counter()-t0,
            "input_tokens": answer["input_tokens"], "output_tokens": answer["output_tokens"]})
        if (i+1) % 25 == 0:
            print(f"Evaluated {i+1}/{len(rows)}", flush=True)
    result = metrics([r["truth"] for r in predictions], [r["prediction"] for r in predictions])
    result["elapsed_seconds"] = time.perf_counter()-started
    output = Path(output)
    write_jsonl(output/"predictions.jsonl", predictions)
    write_jsonl(output/"telemetry.jsonl", telemetry)
    save_json(output/"metrics.json", result)
    return result

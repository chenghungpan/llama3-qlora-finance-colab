"""One-process baseline -> QLoRA -> test evaluation; no second base model in VRAM."""
import argparse
import hashlib
import importlib.metadata
import json
import platform
import time
from pathlib import Path
from .core import (MODEL_ID, MODEL_REVISION, PROMPT_VERSION, encode_example,
                   read_jsonl, save_json, subset, write_jsonl)

def execute(config_path, data_dir, output, resume=False, hourly_usd=None):
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import Trainer, TrainingArguments, set_seed
    from transformers.trainer_utils import get_last_checkpoint
    from .baselines import evaluate_baselines
    from .modeling import CompletionCollator, evaluate_model, load_quantized

    config = json.loads(Path(config_path).read_text())
    data_dir, output = Path(data_dir), Path(output)
    splits = {name: subset(read_jsonl(data_dir/f"{name}.jsonl"), config[f"{name}_limit"], config["seed"])
              for name in ("train", "validation", "test")}
    identity = {"config": config, "model": MODEL_ID, "model_revision": MODEL_REVISION,
        "prompt_version": PROMPT_VERSION,
        "split_sha256": {k: hashlib.sha256((data_dir/f"{k}.jsonl").read_bytes()).hexdigest() for k in splits},
        "selected_ids": {k: [r["id"] for r in rows] for k, rows in splits.items()}}
    checkpoint = None
    if output.exists() and any(output.iterdir()):
        if not resume:
            raise ValueError("Output already exists. Choose a new output directory or use --resume.")
        if not (output/"experiment.json").exists() or json.loads((output/"experiment.json").read_text()) != identity:
            raise ValueError("Cannot resume: config, model, prompt, or data changed")
        checkpoint = get_last_checkpoint(str(output/"checkpoints"))
        if checkpoint is None:
            raise ValueError("No checkpoint to resume. Use a fresh output directory.")
    elif resume:
        raise ValueError("Cannot resume a missing run")
    output.mkdir(parents=True, exist_ok=True)
    save_json(output/"experiment.json", identity)
    save_json(output/"dataset_manifest.json", json.loads((data_dir/"manifest.json").read_text()))
    set_seed(config["seed"])
    start = time.perf_counter()
    save_json(output/"environment.json", {"python": platform.python_version(),
        "packages": {p: importlib.metadata.version(p) for p in
            ("torch", "transformers", "peft", "accelerate", "bitsandbytes", "scikit-learn")},
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None})
    if not (output/"cpu_baselines.json").exists():
        save_json(output/"cpu_baselines.json", evaluate_baselines(splits["train"], splits["test"]))
    model, tokenizer, dtype = load_quantized()
    if not (output/"base/metrics.json").exists():
        evaluate_model(model, tokenizer, splits["test"], output/"base", config["max_length"])
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False})
    model = get_peft_model(model, LoraConfig(r=config["lora_r"], lora_alpha=config["lora_alpha"],
        lora_dropout=config["lora_dropout"], target_modules="all-linear", bias="none", task_type="CAUSAL_LM"))
    model.config.use_cache = False
    model.print_trainable_parameters()
    encoded = {name: [encode_example(tokenizer, r, config["max_length"]) for r in splits[name]]
               for name in ("train", "validation")}
    args = TrainingArguments(output_dir=str(output/"checkpoints"),
        per_device_train_batch_size=1, per_device_eval_batch_size=1,
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        learning_rate=config["learning_rate"], num_train_epochs=config["epochs"], max_steps=config["max_steps"],
        warmup_ratio=0.03, lr_scheduler_type="cosine", optim="paged_adamw_8bit",
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=dtype == torch.bfloat16, fp16=dtype == torch.float16,
        eval_strategy="steps", eval_steps=config["save_steps"],
        save_strategy="steps", save_steps=config["save_steps"], save_total_limit=2,
        logging_steps=1, report_to="none", seed=config["seed"], data_seed=config["seed"],
        prediction_loss_only=True, label_names=["labels"], remove_unused_columns=False, dataloader_num_workers=0)
    trainer = Trainer(model=model, args=args, train_dataset=encoded["train"], eval_dataset=encoded["validation"],
        data_collator=CompletionCollator(tokenizer.pad_token_id), processing_class=tokenizer)
    torch.cuda.reset_peak_memory_stats()
    training_start = time.perf_counter()
    result = trainer.train(resume_from_checkpoint=checkpoint)
    training_seconds = time.perf_counter()-training_start
    validation = trainer.evaluate()
    trainer.save_model(str(output/"adapter"))
    tokenizer.save_pretrained(output/"adapter")
    save_json(output/"adapter/tutorial_config.json", identity)
    save_json(output/"training.json", {"train": result.metrics, "validation": validation,
        "training_seconds_this_session": training_seconds,
        "peak_allocated_gib": torch.cuda.max_memory_allocated()/2**30,
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "log_history": trainer.state.log_history})
    model.config.use_cache = True
    model.gradient_checkpointing_disable()
    evaluate_model(model, tokenizer, splits["test"], output/"finetuned", config["max_length"])
    elapsed = time.perf_counter()-start
    save_json(output/"compute.json", {"session_seconds": elapsed, "hourly_usd_assumption": hourly_usd,
        "estimated_session_usd": elapsed/3600*hourly_usd if hourly_usd is not None else None,
        "note": "Estimate only: excludes setup, downloads before this process, idle time, and earlier resume sessions. Colab bills compute units, not a fixed hourly dollar rate."})
    (output/"adapter/README.md").write_text(
        "# Llama 3 financial sentiment tutorial adapter\n\nBuilt with Meta Llama 3.\n\n"
        "Educational sentiment classification; not a price predictor or trading system.\n"
        "Base: "+MODEL_ID+" @ "+MODEL_REVISION+"\n\n"
        "See tutorial_config.json for data selections and training configuration.\n"
        "See ../finetuned/metrics.json for measured results; no performance is guaranteed.\n"
        "Before distribution, include applicable Meta license/notice and dataset attribution.\n")
    from .report import make_report
    make_report(output)
    print(f"Finished. Adapter and measured reports: {output.resolve()}")

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/quick.json")
    p.add_argument("--data", default="data")
    p.add_argument("--output", default="runs/quick")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--hourly-usd", type=float, help="Optional compute cost assumption; unknown by default")
    a = p.parse_args()
    if a.hourly_usd is not None and a.hourly_usd < 0:
        p.error("--hourly-usd cannot be negative")
    execute(a.config, a.data, a.output, a.resume, a.hourly_usd)

# How the experiment works

## Full fine-tuning, LoRA, and QLoRA

Full fine-tuning updates all base-model parameters and needs memory for their gradients and optimizer states. LoRA freezes the base weights and trains low-rank update matrices. QLoRA additionally quantizes the frozen base weights to reduce memory, while keeping adapters trainable. Quantization does not make all training operations 4-bit.

Our path uses NF4, double quantization, a paged 8-bit optimizer, and adapters on linear layers. Rank 8 is a small starting capacity, not a universally optimal setting. See [PEFT quantization guidance](https://huggingface.co/docs/peft/developer_guides/quantization) and the [QLoRA paper](https://arxiv.org/abs/2305.14314).

## Why the base model matters

We intentionally use `meta-llama/Meta-Llama-3-8B`, not a 3.1 model or an Instruct variant. A base completion model may not follow our one-word instruction before training. Invalid output rate therefore measures formatting as well as classification. Do not interpret the entire improvement as new financial reasoning. An optional future comparison with an instruction-tuned model must be reported as a different baseline.

## Tokenization and loss

`prompt_ids` constructs BOS + instruction + source text + answer cue. It truncates only the text. `encode_example` appends the label and EOS; labels are -100 over the prompt. Causal language-model loss shifts logits/labels internally, so the final prompt token predicts the first answer token. The collator masks only padding positions and leaves the genuine answer EOS supervised, even though PAD and EOS share a token ID.

Batch size 1 with accumulation 8 approximates an effective batch of 8 on one GPU. Gradient checkpointing recomputes activations to trade speed for memory. BF16 is used when supported, otherwise FP16. Inference is greedy and single-example to conserve VRAM.

## Evaluation protocol

All four models in a run see the same selected training/test examples where applicable. The pretrained Llama baseline has no task-specific training. TF-IDF fits its vocabulary and classifier on training only. The QLoRA Trainer receives training and validation, never test examples. It saves the final adapter; this tutorial does not claim best-checkpoint selection.

The parser accepts exactly one class word after trimming whitespace and normalizing case. A longer answer is invalid and counts as a classification error. The confusion matrix includes an invalid prediction column, and macro-F1 always averages the three classes.

Small profiles use balanced subsets to expose all labels quickly. Use the full profile for natural class prevalence. A random sentiment split supports only this historical classification comparison. It is not a chronological trading backtest, and source text could have appeared during base-model pretraining.

## Suggested exercises

1. On validation, compare rank 4 versus 8 while keeping other settings fixed.
2. Compare 500, 1,500, and all training examples. Record compute and F1 together.
3. Measure the fraction of truncated texts using the same tokenizer.
4. Add bootstrap confidence intervals to final metrics; do not mistake a small score difference for a robust improvement.
5. Add semantic near-duplicate clustering before splitting and report how counts change.

If you later add synthetic data, generate it from training material only, mark provenance, audit labels, and compare against the real-only run. Never substitute generated examples for the real held-out test.

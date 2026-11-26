# Llama 3 + QLoRA: financial sentiment in Google Colab

Learn to fine-tune **Meta Llama 3 8B** on real financial text, compare it with simpler baselines, and save/reload your own adapter. Built with Meta Llama 3.

This is a standalone educational repository. It has no dependency on any other application. You do not need your own dataset. The notebooks download the public **Twitter Financial News Sentiment** dataset from a pinned Hugging Face revision.

**Task:** classify a financial post as `bearish`, `bullish`, or `neutral`. This project does not predict prices or demonstrate trading profitability.

## Start here

1. Click **Open in Colab** for lesson 1 below and run cells from top to bottom. The first cell downloads this repository automatically; no ZIP upload is needed.
2. Complete lesson 1 on CPU. For lesson 2, select a GPU runtime, obtain [Llama 3 access](https://huggingface.co/meta-llama/Meta-Llama-3-8B), and add `HF_TOKEN` in Colab Secrets.
3. Run `smoke` first. Then select `quick` with a new run name. Save to Google Drive to retain results through runtime resets.
4. Use lesson 3 in a fresh runtime to reload your adapter and inspect errors.

A Google account is needed to run Colab; a GitHub account is not needed to read this public project. For an offline source copy, clone the repository or create a ZIP with `python scripts/package_repo.py`. The notebooks retain a ZIP-upload option if you clear `REPO_URL`.

| Lesson | Run | Hardware |
|---|---|---|
| [01: Data and baselines](notebooks/01_data_and_baselines.ipynb) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/chenghungpan/llama3-qlora-finance-colab/blob/main/notebooks/01_data_and_baselines.ipynb) | CPU |
| [02: QLoRA training](notebooks/02_qlora_training.ipynb) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/chenghungpan/llama3-qlora-finance-colab/blob/main/notebooks/02_qlora_training.ipynb) | NVIDIA GPU |
| [03: Reload and error analysis](notebooks/03_reload_and_error_analysis.ipynb) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/chenghungpan/llama3-qlora-finance-colab/blob/main/notebooks/03_reload_and_error_analysis.ipynb) | GPU for inference |

## What is included

- Pinned base model and data revisions, SHA-256 data provenance, fixed seeds, and saved example selections.
- A shared versioned completion prompt for training and generation; no chat template is needed for the base model.
- Frozen 4-bit NF4 base weights, double quantization, rank-8 adapters on linear layers, gradient checkpointing, and paged 8-bit AdamW.
- Completion-only loss: train on the answer and EOS, mask prompt/padding, preserve the answer when truncating.
- Majority, TF-IDF/logistic regression, base Llama, and QLoRA comparisons on identical examples.
- Accuracy, macro-F1, per-class precision/recall, confusion matrix, invalid output rate, charts, latency/token counts, GPU memory, and optional compute-cost estimates.
- Drive checkpoints, resume validation, adapter export, fresh-session reload, CPU tests, and GitHub Actions.

The tutorial uses Transformers `Trainer` and PEFT directly rather than adding TRL. This keeps the supervised training loop and loss masking visible without an additional version-sensitive abstraction.

## Data and evaluation

Source: [zeroshot/twitter-financial-news-sentiment](https://huggingface.co/datasets/zeroshot/twitter-financial-news-sentiment), revision `ccbe24de388e287beb92dd393a335c376b350ac3`. The publisher declares MIT licensing. Labels map exactly as upstream: **0 bearish, 1 bullish, 2 neutral**.

The actual pinned files have **9,543 train rows and 2,388 upstream validation rows**. These differ from numbers in the dataset card; the downloader reports actual counts. We use upstream validation as final test, remove normalized exact duplicates/conflicts before splitting, and reserve 10% of cleaned upstream training for validation.

Verified cleaned sizes: **8,583 train / 953 validation / 2,388 test**. Seven duplicate rows are removed, including one train/test overlap. No conflicting-label duplicate group was observed in this revision. Near-duplicates and pretraining contamination remain possible.

Smoke and quick select balanced subsets for teaching. Their class prevalence differs from the full dataset; compare models within the same run, not raw scores across profiles. The full profile preserves the natural class distribution. Validation drives development. The final test is for reporting, not repeated hyperparameter selection.

## Hardware, runtime, and cost

Use a single NVIDIA GPU with about 16 GB or more VRAM and at least 25 GiB free disk after installation. The conservative settings target Colab T4-class hardware, but **the full 8B quantized GPU path has not been executed in the authoring environment**. Do not interpret the memory target as a measured guarantee. Run smoke first. [Colab availability and limits vary](https://research.google.com/colaboratory/faq.html).

| Profile | Training examples | Validation/test | Training length |
|---|---:|---|---|
| smoke | 96 | 24 / 30 | 3 optimizer steps; verifies wiring only |
| quick | 1,500 | 150 / 150 | 1 epoch |
| full | all cleaned | all cleaned | 2 epochs |

All profiles use sequence length 256, batch size 1, gradient accumulation 8, rank 8, alpha 16, and learning rate 2e-4. Runtime depends on GPU, dataset size, downloads, and Colab allocation; no fixed duration or improvement is promised.

There is no paid model API. GPU compute can still cost money. Reports measure elapsed time and leave dollar cost unknown unless you supply `--hourly-usd` as an explicit assumption. This is not a Colab billing quote. Disconnect/delete the runtime after use.

## Local CPU workflow

Requires Python 3.10–3.13; Python 3.11 or 3.12 is recommended for these package pins.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-cpu.txt -e .
python -m unittest discover -s tests -v
python scripts/validate_notebooks.py
python -m finance_qlora.data
```

Tests work without credentials or network. ML integration tests skip when their optional libraries are absent. An additional tiny random Llama test exercises real Trainer/PEFT training and adapter round-trip when the ML libraries are installed; it is not evidence that 8B QLoRA fits a GPU.

## Local NVIDIA GPU workflow

Use a fresh Python environment; do not install into an unrelated project's environment. Set `HF_TOKEN` securely in your environment (never commit it), with approved model access. Install the GPU requirements, then:

```bash
python -m pip install -r requirements-colab.txt -e .
python -m finance_qlora.preflight
python -m finance_qlora.data
python -m finance_qlora.run --config configs/smoke.json --output runs/smoke-01
python -m finance_qlora.run --config configs/quick.json --output runs/quick-01
python -m finance_qlora.predict --adapter runs/quick-01/adapter \
  --text "The company raised its earnings outlook after stronger sales."
```

To resume an interrupted run after a checkpoint was saved:

```bash
python -m finance_qlora.run --config configs/quick.json --output runs/quick-01 --resume
```

Keep the config and data unchanged. A new experiment requires a new output directory. Checkpoints are saved every 25 optimizer steps, keeping the latest two. A three-step smoke run saves its final adapter but does not produce a resumable optimizer checkpoint.

## Outputs

```text
runs/quick-01/
  adapter/                 # LoRA weights, tokenizer, exact tutorial config
  checkpoints/             # Optimizer/RNG state for resuming
  base/                    # Untuned predictions, metrics, token/latency telemetry
  finetuned/               # Adapter predictions, metrics, token/latency telemetry
  experiment.json          # Configuration, model revision, selected IDs, data hashes
  dataset_manifest.json    # Data provenance, counts, cleaning audit
  environment.json         # Python, key package versions, GPU model
  training.json            # Loss history, validation loss, peak memory
  compute.json             # Session time and optional cost estimate
  cpu_baselines.json
  comparison.json
  report.md
  comparison.png
  confusion.png
```

Do not commit tokens, downloaded posts, checkpoints, or adapters. The `.gitignore` excludes default data/run directories. Notebooks ship without outputs or secrets. Artifacts are saved locally/Drive only; the code never pushes to the Hub.

## Continue learning

- [Concepts and design choices](docs/concepts.md)
- [Troubleshooting](docs/troubleshooting.md)
- [Forecasting extension research design](docs/forecasting_extension.md)
- [Data/model/code licensing](docs/licenses.md)
- [Validation evidence and limitations](docs/validation.md)

This implementation teaches an experiment, not a guaranteed performance improvement. A small TF-IDF model may be the better solution for this classification task.

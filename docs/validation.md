# Validation evidence

Validated during authoring on macOS ARM64 with Python 3.12.2. The source package is independent of the author's other projects.

## Executed checks

- Downloaded the public dataset at the pinned Hugging Face revision and inspected CSV headers and numeric labels.
- Ran preprocessing on both real source files. Result: 8,583 training, 953 validation, and 2,388 test rows, with disjoint normalized text IDs.
- Ran 11 automated tests covering label parsing, invalid predictions, duplicate/conflict removal, stratified splitting, reproducible subsets, prompt truncation, completion-only masking, EOS/padding behavior, CPU baselines, notebook/config structure, and real tiny-model training/save/reload.
- The tiny CPU model uses the actual pinned Torch 2.7.1, Transformers 4.53.3, PEFT 0.16.0, and Accelerate 1.9.0 APIs. It verifies finite training and validation loss, adapter updates, safetensors export, and equal logits after adapter reload onto an identical base.
- Validated all three notebooks using nbformat and parsed their code cells as Python. Notebooks ship without outputs, tokens, or execution state.
- Compiled the Python package and exercised the real-data TF-IDF validation baseline.

## Measured CPU validation results

These scores use all 8,583 cleaned training examples and the 953-example internal validation split, not the final test. The machine-readable results are in `cpu_validation_results.json`.

| Model | Accuracy | Macro-F1 |
|---|---:|---:|
| Majority | 0.6474 | 0.2620 |
| TF-IDF + logistic regression | 0.7943 | 0.6787 |

## Not executed here

The authoring environment has no NVIDIA GPU and no approved Llama model credentials were used. Full 8B weight download, bitsandbytes 4-bit loading, CUDA training, GPU memory fit, Drive checkpoints, and an interactive Colab session remain unverified. The tiny CPU test does not validate quantized CUDA kernels or Colab runtime compatibility.

No Llama baseline or QLoRA accuracy is fabricated. `report.md` and charts for those models are produced only by an actual GPU run. Use the three-step smoke profile first; it checks execution rather than useful learning. After a real run, preserve its environment, configuration, checksums, and metrics alongside this evidence.

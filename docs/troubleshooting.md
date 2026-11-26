# Troubleshooting

| Symptom | Action |
|---|---|
| 401 / 403 / gated model | Accept the exact Llama 3 model license, wait for access, and use a read token with gated-repository permission. Enable Colab Secret access. |
| `SecretNotFoundError` | Add `HF_TOKEN` using Colab's key icon. Do not paste it into code. |
| No GPU / bitsandbytes CUDA error | Select an NVIDIA GPU runtime. Apple MPS and CPU are not supported for this 4-bit training path. |
| CUDA out of memory | Delete the runtime to release stale allocations. Retry smoke. Shorten `max_length` to 192 or 128 in a new config/run; keep the prompt-length validation. Reduce rank to 4. Increasing accumulation does not reduce batch-1 activation memory. |
| Import errors involving torchvision / torchaudio | In a fresh disposable runtime remove these unused libraries, then install the pinned requirements. The GPU notebook does this. |
| Package changes not visible | ML jobs run as fresh subprocesses. If you imported Torch/Transformers directly in notebook cells, restart the runtime and rerun bootstrap/auth. |
| Model download is large/slow | Base weights are downloaded before quantization. Keep at least 25 GiB free disk after environment installation; retries use the Hugging Face cache. |
| Colab disconnects | Save to Drive, rerun setup/auth/data, and set RESUME=True with the same configuration and run directory. |
| No checkpoint to resume | The process ended before the first 25-step save, or it was a smoke run. Start a fresh run name. An adapter alone lacks optimizer/RNG state for exact resume. |
| Run directory already exists | Use a new run name, or resume an interrupted run explicitly. This prevents mixing experiments. |
| Many invalid base-model outputs | The base model is not instruction-tuned. Inspect output format separately from class accuracy. |
| Fine-tuned model underperforms TF-IDF | This is a valid result. Use validation to investigate training length, labels, class imbalance, and truncation. |
| Test scores look surprisingly strong | Check duplicate leakage and historical pretraining contamination; do not infer trading ability. |
| Dataset download fails | Retry later or use `--source-dir` with the two original CSVs. The manifest marks the override and records actual hashes. |
| Lesson 3 cannot find adapter | Correct RUN_DIR to the Drive folder from lesson 2, not the source repository folder. |

Dependencies are pinned at the direct-package level, not a universal cross-platform lockfile. Colab images and CUDA drivers can change. Save the generated environment metadata with each experiment. No GPU benchmark is included until someone actually executes and records the GPU run.

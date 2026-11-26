"""Fail early before a multi-gigabyte model download."""
import shutil
import sys

def main():
    import torch
    from huggingface_hub import hf_hub_download
    from .core import MODEL_ID, MODEL_REVISION
    if not torch.cuda.is_available():
        raise SystemExit("No NVIDIA GPU: select a GPU runtime in Colab, then rerun setup.")
    print("Python:", sys.version.split()[0])
    print("GPU:", torch.cuda.get_device_name(0))
    memory = torch.cuda.get_device_properties(0).total_memory/2**30
    disk = shutil.disk_usage('.').free/2**30
    print(f"GPU memory: {memory:.1f} GiB; free disk: {disk:.1f} GiB")
    if memory < 14:
        raise SystemExit("Use a GPU with at least 14 GiB visible VRAM for this configuration.")
    if disk < 25:
        raise SystemExit("Allow at least 25 GiB free disk for model weights and checkpoints.")
    try:
        hf_hub_download(MODEL_ID, "config.json", revision=MODEL_REVISION)
    except Exception as exc:
        raise SystemExit("Model access check failed. Accept the Llama 3 license, wait for approval, and set a read HF_TOKEN with access to gated repositories.") from exc
    print("Access check passed. Memory is a preflight estimate, not a guarantee that training fits.")

if __name__ == "__main__":
    main()

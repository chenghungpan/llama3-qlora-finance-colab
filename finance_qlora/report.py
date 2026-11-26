"""Generate readable comparisons using measured results only."""
import argparse
import json
from pathlib import Path
from .core import save_json

def make_report(run_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    run = Path(run_dir)
    results = json.loads((run/"cpu_baselines.json").read_text())
    for name, folder in (("llama3_base", "base"), ("llama3_qlora", "finetuned")):
        results[name] = json.loads((run/folder/"metrics.json").read_text())
    save_json(run/"comparison.json", results)
    lines = ["# Measured results", "", "All models use the same selected test examples. Small balanced subsets are teaching diagnostics, not population performance estimates.", "",
             "| Model | N | Accuracy | Macro-F1 | Invalid outputs |",
             "|---|---:|---:|---:|---:|"]
    for name, m in results.items():
        lines.append(f"| {name} | {m['n']} | {m['accuracy']:.4f} | {m['macro_f1']:.4f} | {m['invalid_rate']:.4f} |")
    lines += ["", "Do not tune on this test set. Use training/validation for new experiments.",
        "A worse QLoRA result is still a valid finding. Check class errors, format failures, and the simpler baseline.",
        "", "Sentiment labels are not realized stock returns. Historical public data may overlap model pretraining.", ""]
    (run/"report.md").write_text("\n".join(lines))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(list(results), [r["macro_f1"] for r in results.values()], color=["#94a3b8", "#64748b", "#0ea5e9", "#0d9488"])
    ax.set(ylabel="Macro-F1", ylim=(0, 1), title="Held-out financial sentiment classification")
    fig.tight_layout(); fig.savefig(run/"comparison.png", dpi=160); plt.close(fig)
    m = results["llama3_qlora"]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.imshow(m["confusion_matrix"], cmap="Blues")
    ax.set(xticks=range(4), xticklabels=m["column_labels"], yticks=range(3), yticklabels=m["row_labels"],
           xlabel="Predicted", ylabel="True", title="QLoRA confusion matrix")
    for i, row in enumerate(m["confusion_matrix"]):
        for j, count in enumerate(row):
            ax.text(j, i, str(count), ha="center", va="center", color="black")
    fig.tight_layout(); fig.savefig(run/"confusion.png", dpi=160); plt.close(fig)
    return results

if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("run_dir"); make_report(p.parse_args().run_dir)

"""
Compile evaluation results into tables and comparison plots.
Compares Zero-Shot (Grounding DINO + SAM) vs SegFormer-B0 vs SegFormer-B2.
"""
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

RESULTS_DIR = Path("results")
FIGURES_DIR = RESULTS_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ── Load metrics ────────────────────────────────────────────────────────
with open(RESULTS_DIR / "pre_annotation" / "pre_annotation_metrics_dataset_level.json") as f:
    zero_shot = json.load(f)

with open(RESULTS_DIR / "segformer" / "models_comparison_summary.json") as f:
    segformer = json.load(f)

b0 = segformer["b0"]
b2 = segformer["b2"]

classes = [
    "background", "building", "window", "sky", "roof",
    "door", "tree", "people", "car", "sign"
]

rows = []
for c in classes:
    zs_val = zero_shot["iou_per_class"].get(c, 0.0) * 100
    b0_val = b0["per_class_iou"].get(c, 0.0) * 100
    b2_val = b2["per_class_iou"].get(c, 0.0) * 100
    gain_b2_zs = b2_val - zs_val
    gain_b2_b0 = b2_val - b0_val
    rows.append({
        "class": c,
        "zero_shot": zs_val,
        "b0": b0_val,
        "b2": b2_val,
        "gain_b2_zs": gain_b2_zs,
        "gain_b2_b0": gain_b2_b0,
    })

summary_stats = {
    "miou_10": {
        "zero_shot": zero_shot["miou_10"] * 100,
        "b0": b0["miou_10"] * 100,
        "b2": b2["miou_10"] * 100,
        "gain_b2_zs": (b2["miou_10"] - zero_shot["miou_10"]) * 100,
        "gain_b2_b0": (b2["miou_10"] - b0["miou_10"]) * 100,
    },
    "miou_9_no_bg": {
        "zero_shot": zero_shot["miou_9_no_bg"] * 100,
        "b0": b0["miou_9_no_bg"] * 100,
        "b2": b2["miou_9_no_bg"] * 100,
        "gain_b2_zs": (b2["miou_9_no_bg"] - zero_shot["miou_9_no_bg"]) * 100,
        "gain_b2_b0": (b2["miou_9_no_bg"] - b0["miou_9_no_bg"]) * 100,
    }
}

# ── Save JSON Table ──────────────────────────────────────────────────────
table_output = {
    "per_class": rows,
    "summary": summary_stats,
}
with open(RESULTS_DIR / "model_comparison_table.json", "w") as f:
    json.dump(table_output, f, indent=2)

# ── 1. Bar Chart: Per-Class IoU Comparison ──────────────────────────────
x = np.arange(len(classes))
width = 0.26

fig, ax = plt.subplots(figsize=(14, 6))
rects1 = ax.bar(x - width, [r["zero_shot"] for r in rows], width, label="Zero-Shot (Grounding DINO + SAM)", color="#4575b4")
rects2 = ax.bar(x, [r["b0"] for r in rows], width, label="SegFormer-B0", color="#fdae61")
rects3 = ax.bar(x + width, [r["b2"] for r in rows], width, label="SegFormer-B2", color="#d73027")

ax.set_ylabel("Dataset-level IoU (%)", fontsize=12)
ax.set_title("Façade Parsing Performance: Zero-Shot vs Fine-Tuned SegFormer (CFP Test Set)", fontsize=13, pad=15)
ax.set_xticks(x)
ax.set_xticklabels(classes, fontsize=11, rotation=20)
ax.legend(fontsize=11)
ax.grid(axis="y", linestyle="--", alpha=0.4)
ax.set_ylim(0, 105)

# Add values on top of bars
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f"{height:.1f}",
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center", va="bottom", fontsize=8)

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)

plt.tight_layout()
plt.savefig(FIGURES_DIR / "model_comparison_iou.png", dpi=150)
plt.close()

# ── 2. Training Curves: B0 vs B2 ─────────────────────────────────────────
history_b0_path = RESULTS_DIR / "segformer" / "segformer_b0" / "train_history.json"
history_b2_path = RESULTS_DIR / "segformer" / "segformer_b2" / "train_history.json"

if history_b0_path.exists() and history_b2_path.exists():
    with open(history_b0_path) as f:
        hist_b0 = json.load(f)
    with open(history_b2_path) as f:
        hist_b2 = json.load(f)

    epochs_b0 = [h["epoch"] for h in hist_b0]
    epochs_b2 = [h["epoch"] for h in hist_b2]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss curve
    ax1.plot(epochs_b0, [h["train_loss"] for h in hist_b0], label="B0 Train", color="#fdae61", linestyle="--")
    ax1.plot(epochs_b0, [h["val_loss"] for h in hist_b0], label="B0 Val", color="#e66101")
    ax1.plot(epochs_b2, [h["train_loss"] for h in hist_b2], label="B2 Train", color="#92c5de", linestyle="--")
    ax1.plot(epochs_b2, [h["val_loss"] for h in hist_b2], label="B2 Val", color="#0571b0")
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Cross-Entropy Loss", fontsize=11)
    ax1.set_title("Training & Validation Loss", fontsize=12)
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Validation mIoU curve
    ax2.plot(epochs_b0, [h["val_miou_9_no_bg"] * 100 for h in hist_b0], label="B0 Val mIoU (no bg)", color="#e66101", linewidth=2)
    ax2.plot(epochs_b2, [h["val_miou_9_no_bg"] * 100 for h in hist_b2], label="B2 Val mIoU (no bg)", color="#0571b0", linewidth=2)
    ax2.axhline(zero_shot["miou_9_no_bg"] * 100, color="gray", linestyle=":", label=f"Zero-shot baseline ({zero_shot['miou_9_no_bg']*100:.1f}%)")
    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("mIoU (no bg) (%)", fontsize=11)
    ax2.set_title("Validation mIoU Convergence", fontsize=12)
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "training_curves.png", dpi=150)
    plt.close()

print("Evaluation compilation complete:")
print(f"  - Table saved to {RESULTS_DIR / 'model_comparison_table.json'}")
print(f"  - Bar chart saved to {FIGURES_DIR / 'model_comparison_iou.png'}")
print(f"  - Training curves saved to {FIGURES_DIR / 'training_curves.png'}")


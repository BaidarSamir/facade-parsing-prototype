"""
Generate final figures and tables from full-image test evaluation.
1. Updated bar chart: Zero-Shot vs SegFormer-B0 vs SegFormer-B2 (results/figures/model_comparison_iou.png)
2. Row-normalized SegFormer-B2 confusion matrix (results/figures/b2_confusion_matrix.png)
3. Final comparison table (results/model_comparison_table.json)
"""
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

RESULTS_DIR = Path("results")
FIGURES_DIR = RESULTS_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ── 1. Load Data ─────────────────────────────────────────────────────────
with open(RESULTS_DIR / "pre_annotation" / "pre_annotation_metrics_dataset_level.json") as f:
    zero_shot = json.load(f)

with open(RESULTS_DIR / "segformer" / "full_image_results.json") as f:
    full_results = json.load(f)

crop_ref_path = RESULTS_DIR / "segformer" / "models_comparison_summary.json"
crop_ref = {}
if crop_ref_path.exists():
    with open(crop_ref_path) as f:
        crop_ref = json.load(f)

classes = [
    "background", "building", "window", "sky", "roof",
    "door", "tree", "people", "car", "sign"
]

b0_full = full_results["b0"]
b2_full = full_results["b2"]

# ── 2. Build Comparison Rows ─────────────────────────────────────────────
rows = []
for c in classes:
    zs_val = zero_shot["iou_per_class"].get(c, 0.0) * 100
    b0_val = b0_full["per_class_iou"].get(c, 0.0)
    b2_val = b2_full["per_class_iou"].get(c, 0.0)
    gain_b2_zs = b2_val - zs_val
    gain_b2_b0 = b2_val - b0_val
    rows.append({
        "class": c,
        "zero_shot": round(zs_val, 2),
        "b0_full": round(b0_val, 2),
        "b2_full": round(b2_val, 2),
        "gain_b2_zs": round(gain_b2_zs, 2),
        "gain_b2_b0": round(gain_b2_b0, 2),
    })

comparison_table = {
    "evaluation_mode": "full_image_longest_side_1024",
    "headline_summary": {
        "miou_10_classes": {
            "zero_shot": round(zero_shot["miou_10"] * 100, 2),
            "segformer_b0": round(b0_full["miou_10"], 2),
            "segformer_b2": round(b2_full["miou_10"], 2),
            "gain_b2_zs": round(b2_full["miou_10"] - zero_shot["miou_10"] * 100, 2),
            "gain_b2_b0": round(b2_full["miou_10"] - b0_full["miou_10"], 2),
        },
        "miou_9_classes_no_bg": {
            "zero_shot": round(zero_shot["miou_9_no_bg"] * 100, 2),
            "segformer_b0": round(b0_full["miou_9_no_bg"], 2),
            "segformer_b2": round(b2_full["miou_9_no_bg"], 2),
            "gain_b2_zs": round(b2_full["miou_9_no_bg"] - zero_shot["miou_9_no_bg"] * 100, 2),
            "gain_b2_b0": round(b2_full["miou_9_no_bg"] - b0_full["miou_9_no_bg"], 2),
        }
    },
    "per_class": rows,
    "note_center_crop_512_reference": {
        "b0_miou_10": round(crop_ref.get("b0", {}).get("miou_10", 0.0) * 100, 2),
        "b0_miou_9_no_bg": round(crop_ref.get("b0", {}).get("miou_9_no_bg", 0.0) * 100, 2),
        "b2_miou_10": round(crop_ref.get("b2", {}).get("miou_10", 0.0) * 100, 2),
        "b2_miou_9_no_bg": round(crop_ref.get("b2", {}).get("miou_9_no_bg", 0.0) * 100, 2),
        "explanation": "Evaluated on 512x512 center crops during training validation epochs."
    }
}

with open(RESULTS_DIR / "model_comparison_table.json", "w") as f:
    json.dump(comparison_table, f, indent=2)

# ── 3. Bar Chart: Per-Class IoU (Full Image) ─────────────────────────────
x = np.arange(len(classes))
width = 0.26

fig, ax = plt.subplots(figsize=(14, 6))
rects1 = ax.bar(x - width, [r["zero_shot"] for r in rows], width, label="Zero-Shot (Grounding DINO + SAM)", color="#4575b4")
rects2 = ax.bar(x, [r["b0_full"] for r in rows], width, label="SegFormer-B0 (Full Image)", color="#fdae61")
rects3 = ax.bar(x + width, [r["b2_full"] for r in rows], width, label="SegFormer-B2 (Full Image)", color="#d73027")

ax.set_ylabel("Dataset-level IoU (%)", fontsize=12)
ax.set_title("Façade Parsing Performance: Full-Image Test Set Evaluation (CFP)", fontsize=13, pad=15)
ax.set_xticks(x)
ax.set_xticklabels(classes, fontsize=11, rotation=20)
ax.legend(fontsize=11)
ax.grid(axis="y", linestyle="--", alpha=0.4)
ax.set_ylim(0, 105)

for i in range(len(classes)):
    ax.annotate(f"{rows[i]['zero_shot']:.1f}", xy=(x[i] - width, rows[i]['zero_shot']), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8)
    ax.annotate(f"{rows[i]['b0_full']:.1f}", xy=(x[i], rows[i]['b0_full']), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8)
    ax.annotate(f"{rows[i]['b2_full']:.1f}", xy=(x[i] + width, rows[i]['b2_full']), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8)

plt.tight_layout()
bar_path = FIGURES_DIR / "model_comparison_iou.png"
plt.savefig(bar_path, dpi=150)
plt.close()

# ── 4. Confusion Matrix Heatmap from confusion_b2_full.npy ───────────────
cm_path_npy = RESULTS_DIR / "segformer" / "confusion_b2_full.npy"
if cm_path_npy.exists():
    cm = np.load(cm_path_npy)
    row_sums = cm.sum(axis=1, keepdims=True)
    norm_cm = np.divide(cm.astype(np.float64), row_sums, out=np.zeros_like(cm, dtype=np.float64), where=row_sums > 0)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(norm_cm, cmap="Blues", vmin=0, vmax=1.0)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(np.arange(len(classes)))
    ax.set_yticks(np.arange(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right", fontsize=10)
    ax.set_yticklabels(classes, fontsize=10)
    ax.set_xlabel("Predicted Class", fontsize=11, labelpad=10)
    ax.set_ylabel("Ground Truth Class", fontsize=11, labelpad=10)
    ax.set_title("SegFormer-B2 Row-Normalized Confusion Matrix (Recall % on Full Test Set)", fontsize=13, pad=15)

    for r in range(len(classes)):
        for c in range(len(classes)):
            val = norm_cm[r, c] * 100
            color = "white" if norm_cm[r, c] > 0.55 else "black"
            if val >= 0.5:
                ax.text(c, r, f"{val:.1f}%", ha="center", va="center", color=color, fontsize=8)

    plt.tight_layout()
    cm_plot_path = FIGURES_DIR / "b2_confusion_matrix.png"
    plt.savefig(cm_plot_path, dpi=150)
    plt.close()

print("✓ All final figures and tables successfully generated:")
print(f"  - Table: {RESULTS_DIR / 'model_comparison_table.json'}")
print(f"  - Bar Chart: {bar_path}")
if cm_path_npy.exists():
    print(f"  - Confusion Matrix: {FIGURES_DIR / 'b2_confusion_matrix.png'}")


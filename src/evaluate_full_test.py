"""
Full-Image Test Evaluation for SegFormer-B0 and SegFormer-B2 on CFP Test Set.
Evaluates on full resized images (longest side 1024, matching zero-shot protocol).
Outputs:
1. Row-normalized confusion matrix for B2 (results/figures/b2_confusion_matrix.png)
2. 8-image prediction overlay grid (results/figures/b2_test_predictions_grid.png)
3. Updated comparison bar chart (results/figures/model_comparison_iou.png)
4. Full summary table (results/model_comparison_table.json)
"""
import json
import random
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import torch
import torch.nn.functional as F
from torchvision import transforms as T
from transformers import SegformerForSemanticSegmentation, AutoConfig

from config import CLASS_NAMES, CLASS_COLORS_RGB, NUM_CLASSES, DATA_PROCESSED, RESULTS_DIR

FIGURES_DIR = RESULTS_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def remap_ckpt_keys(ckpt_dict, target_keys):
    if set(ckpt_dict.keys()) == set(target_keys):
        return ckpt_dict
    new_dict = {}
    for k, v in ckpt_dict.items():
        k_new = k
        if "segformer.stages." in k_new:
            parts = k_new.split(".")
            stage_idx = parts[2]
            if parts[3] == "patch_embeddings":
                k_new = f"segformer.encoder.patch_embeddings.{stage_idx}." + ".".join(parts[4:])
            elif parts[3] == "blocks":
                block_idx = parts[4]
                sub = ".".join(parts[5:])
                sub = sub.replace("layernorm_before", "layer_norm_1")
                sub = sub.replace("layernorm_after", "layer_norm_2")
                sub = sub.replace("attention.q_proj", "attention.self.query")
                sub = sub.replace("attention.k_proj", "attention.self.key")
                sub = sub.replace("attention.v_proj", "attention.self.value")
                sub = sub.replace("attention.o_proj", "attention.output.dense")
                sub = sub.replace("attention.sequence_reduction.sequence_reduction", "attention.self.sr")
                sub = sub.replace("attention.sequence_reduction.layer_norm", "attention.self.layer_norm")
                sub = sub.replace("mlp.fc1", "mlp.dense1")
                sub = sub.replace("mlp.fc2", "mlp.dense2")
                k_new = f"segformer.encoder.block.{stage_idx}.{block_idx}.{sub}"
            elif parts[3] == "layer_norm":
                k_new = f"segformer.encoder.layer_norm.{stage_idx}." + ".".join(parts[4:])
        k_new = k_new.replace("decode_head.linear_projections.", "decode_head.linear_c.")
        new_dict[k_new] = v
    return new_dict


def load_segformer_model(variant="b2", checkpoint_path=None, device="cpu"):
    pretrained_id = f"nvidia/segformer-{variant.lower()}-finetuned-ade-512-512"
    cfg = AutoConfig.from_pretrained(pretrained_id)
    cfg.num_labels = NUM_CLASSES
    cfg.id2label = {i: name for i, name in CLASS_NAMES.items()}
    cfg.label2id = {name: i for i, name in CLASS_NAMES.items()}
    model = SegformerForSemanticSegmentation(cfg)

    ckpt = torch.load(checkpoint_path, map_location=device)
    remapped = remap_ckpt_keys(ckpt, model.state_dict().keys())
    model.load_state_dict(remapped)
    model.to(device)
    model.eval()
    return model


def predict_full_image(model, img: Image.Image, device="cpu"):
    w, h = img.size
    pad_w = (32 - (w % 32)) % 32
    pad_h = (32 - (h % 32)) % 32
    if pad_w > 0 or pad_h > 0:
        padded = Image.fromarray(np.pad(np.array(img), ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect"))
    else:
        padded = img

    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    t_img = normalize(T.ToTensor()(padded)).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(pixel_values=t_img)
        logits = F.interpolate(out.logits, size=(padded.size[1], padded.size[0]), mode="bilinear", align_corners=False)
        pred = logits.argmax(dim=1).squeeze(0).cpu().numpy()

    return pred[:h, :w]


def mask_to_color(mask):
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for cid, color in CLASS_COLORS_RGB.items():
        rgb[mask == cid] = color
    return rgb


def evaluate_model_on_test(model, test_img_paths, test_msk_paths, device="cpu"):
    total_inter = np.zeros(NUM_CLASSES, dtype=np.float64)
    total_union = np.zeros(NUM_CLASSES, dtype=np.float64)
    confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    preds_dict = {}

    for i, (ip, mp) in enumerate(zip(test_img_paths, test_msk_paths)):
        img = Image.open(ip).convert("RGB")
        gt = np.array(Image.open(mp))
        pred = predict_full_image(model, img, device=device)
        preds_dict[ip.stem] = pred

        valid = (gt >= 0) & (gt < NUM_CLASSES)
        np.add.at(confusion, (gt[valid], pred[valid]), 1)

        for c in range(NUM_CLASSES):
            p_c = pred == c
            g_c = gt == c
            total_inter[c] += np.logical_and(p_c, g_c).sum()
            total_union[c] += np.logical_or(p_c, g_c).sum()

    ious = {}
    valid_10, valid_9 = [], []
    for c in range(NUM_CLASSES):
        name = CLASS_NAMES[c]
        u = total_union[c]
        if u > 0:
            val = float(total_inter[c] / u)
            ious[name] = val
            valid_10.append(val)
            if c > 0:
                valid_9.append(val)
        else:
            ious[name] = None

    miou_10 = float(np.mean(valid_10)) if valid_10 else 0.0
    miou_9 = float(np.mean(valid_9)) if valid_9 else 0.0
    return ious, miou_10, miou_9, confusion, preds_dict


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running full-image test evaluation on: {device}")

    test_img_paths = sorted((DATA_PROCESSED / "test" / "images").glob("*.png"))
    test_msk_paths = sorted((DATA_PROCESSED / "test" / "masks").glob("*.png"))
    print(f"Total test images: {len(test_img_paths)}")

    # 1. Load Zero-Shot results (already on full resized images)
    with open(RESULTS_DIR / "pre_annotation" / "pre_annotation_metrics_dataset_level.json") as f:
        zero_shot = json.load(f)

    # 2. Evaluate SegFormer-B0 on full images
    print("\n--- Evaluating SegFormer-B0 (Full Image) ---")
    model_b0 = load_segformer_model("b0", "models/segformer_b0_best.pt", device=device)
    b0_ious, b0_m10, b0_m9, _, _ = evaluate_model_on_test(model_b0, test_img_paths, test_msk_paths, device=device)
    print(f"SegFormer-B0 Full-Image mIoU (10 classes): {b0_m10*100:.2f}% | (no bg): {b0_m9*100:.2f}%")
    del model_b0

    # 3. Evaluate SegFormer-B2 on full images
    print("\n--- Evaluating SegFormer-B2 (Full Image) ---")
    model_b2 = load_segformer_model("b2", "models/segformer_b2_best.pt", device=device)
    b2_ious, b2_m10, b2_m9, b2_cm, b2_preds = evaluate_model_on_test(model_b2, test_img_paths, test_msk_paths, device=device)
    print(f"SegFormer-B2 Full-Image mIoU (10 classes): {b2_m10*100:.2f}% | (no bg): {b2_m9*100:.2f}%")

    # ── Load center-crop reference numbers for note ─────────────────────────
    center_crop_ref = {}
    ref_file = RESULTS_DIR / "segformer" / "models_comparison_summary.json"
    if ref_file.exists():
        with open(ref_file) as f:
            center_crop_ref = json.load(f)

    # ── Save updated JSON comparison table ──────────────────────────────────
    classes = [CLASS_NAMES[c] for c in range(NUM_CLASSES)]
    rows = []
    for c in classes:
        zs_val = zero_shot["iou_per_class"].get(c, 0.0) * 100
        b0_val = b0_ious.get(c, 0.0) * 100
        b2_val = b2_ious.get(c, 0.0) * 100
        rows.append({
            "class": c,
            "zero_shot_full": zs_val,
            "b0_full": b0_val,
            "b2_full": b2_val,
            "gain_b2_zs": b2_val - zs_val,
            "gain_b2_b0": b2_val - b0_val,
        })

    summary_data = {
        "evaluation_protocol": "full_image_longest_1024",
        "zero_shot_grounding_dino_sam": {
            "miou_10": zero_shot["miou_10"] * 100,
            "miou_9_no_bg": zero_shot["miou_9_no_bg"] * 100,
            "per_class_iou": {k: v * 100 for k, v in zero_shot["iou_per_class"].items()},
        },
        "segformer_b0_full": {
            "miou_10": b0_m10 * 100,
            "miou_9_no_bg": b0_m9 * 100,
            "per_class_iou": {k: v * 100 for k, v in b0_ious.items()},
        },
        "segformer_b2_full": {
            "miou_10": b2_m10 * 100,
            "miou_9_no_bg": b2_m9 * 100,
            "per_class_iou": {k: v * 100 for k, v in b2_ious.items()},
        },
        "center_crop_512_reference": center_crop_ref,
        "per_class_table": rows,
    }

    with open(RESULTS_DIR / "model_comparison_table.json", "w") as f:
        json.dump(summary_data, f, indent=2)
    print(f"\n✓ Updated summary saved to {RESULTS_DIR / 'model_comparison_table.json'}")

    # ── 4. Confusion Matrix Heatmap (B2 Row-Normalized) ─────────────────────
    row_sums = b2_cm.sum(axis=1, keepdims=True)
    norm_cm = np.divide(b2_cm.astype(np.float64), row_sums, out=np.zeros_like(b2_cm, dtype=np.float64), where=row_sums > 0)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(norm_cm, cmap="Blues", vmin=0, vmax=1.0)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(np.arange(NUM_CLASSES))
    ax.set_yticks(np.arange(NUM_CLASSES))
    ax.set_xticklabels(classes, rotation=45, ha="right", fontsize=10)
    ax.set_yticklabels(classes, fontsize=10)
    ax.set_xlabel("Predicted Class", fontsize=11, labelpad=10)
    ax.set_ylabel("Ground Truth Class", fontsize=11, labelpad=10)
    ax.set_title("SegFormer-B2 Row-Normalized Confusion Matrix (Recall % on Full Test Set)", fontsize=13, pad=15)

    for r in range(NUM_CLASSES):
        for c in range(NUM_CLASSES):
            val = norm_cm[r, c] * 100
            color = "white" if norm_cm[r, c] > 0.55 else "black"
            if val >= 0.5:
                ax.text(c, r, f"{val:.1f}%", ha="center", va="center", color=color, fontsize=8)

    plt.tight_layout()
    cm_path = FIGURES_DIR / "b2_confusion_matrix.png"
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"✓ Confusion matrix saved to {cm_path}")

    # ── 5. Overlay Grid (8 Images) ──────────────────────────────────────────
    random.seed(42)
    sample_indices = random.sample(range(len(test_img_paths)), 8)

    fig, axes = plt.subplots(8, 3, figsize=(15, 3.2 * 8))
    for row_idx, idx in enumerate(sample_indices):
        ip = test_img_paths[idx]
        stem = ip.stem
        img = np.array(Image.open(ip))
        gt = np.array(Image.open(test_msk_paths[idx]))
        pred = b2_preds[stem]

        axes[row_idx, 0].imshow(img)
        axes[row_idx, 0].set_title(f"{stem} (Image)", fontsize=9)
        axes[row_idx, 0].axis("off")

        axes[row_idx, 1].imshow(img)
        axes[row_idx, 1].imshow(mask_to_color(gt), alpha=0.55)
        axes[row_idx, 1].set_title("Ground Truth", fontsize=9)
        axes[row_idx, 1].axis("off")

        axes[row_idx, 2].imshow(img)
        axes[row_idx, 2].imshow(mask_to_color(pred), alpha=0.55)
        axes[row_idx, 2].set_title("SegFormer-B2 Prediction (Full Image)", fontsize=9)
        axes[row_idx, 2].axis("off")

    patches = [mpatches.Patch(color=np.array(CLASS_COLORS_RGB[c]) / 255.0, label=f"{c}: {CLASS_NAMES[c]}") for c in range(NUM_CLASSES)]
    fig.legend(handles=patches, loc="lower center", ncol=5, fontsize=10, bbox_to_anchor=(0.5, 0.005))
    plt.suptitle("SegFormer-B2 Test Set Predictions vs Ground Truth (Full-Image)", fontsize=14, y=0.995)
    plt.tight_layout(rect=[0, 0.03, 1, 0.99])

    grid_path = FIGURES_DIR / "b2_test_predictions_grid.png"
    plt.savefig(grid_path, dpi=120)
    plt.close()
    print(f"✓ 8-image overlay grid saved to {grid_path}")

    # ── 6. Updated Bar Chart ────────────────────────────────────────────────
    x = np.arange(len(classes))
    width = 0.26

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(x - width, [r["zero_shot_full"] for r in rows], width, label="Zero-Shot (Grounding DINO + SAM)", color="#4575b4")
    ax.bar(x, [r["b0_full"] for r in rows], width, label="SegFormer-B0 (Full Image)", color="#fdae61")
    ax.bar(x + width, [r["b2_full"] for r in rows], width, label="SegFormer-B2 (Full Image)", color="#d73027")

    ax.set_ylabel("Dataset-level IoU (%)", fontsize=12)
    ax.set_title("Façade Parsing Performance: Full-Image Test Set Evaluation (CFP)", fontsize=13, pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(classes, fontsize=11, rotation=20)
    ax.legend(fontsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_ylim(0, 105)

    for i in range(len(classes)):
        ax.annotate(f"{rows[i]['zero_shot_full']:.1f}", xy=(x[i] - width, rows[i]['zero_shot_full']), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8)
        ax.annotate(f"{rows[i]['b0_full']:.1f}", xy=(x[i], rows[i]['b0_full']), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8)
        ax.annotate(f"{rows[i]['b2_full']:.1f}", xy=(x[i] + width, rows[i]['b2_full']), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8)

    plt.tight_layout()
    bar_path = FIGURES_DIR / "model_comparison_iou.png"
    plt.savefig(bar_path, dpi=150)
    plt.close()
    print(f"✓ Updated bar chart saved to {bar_path}")


if __name__ == "__main__":
    main()


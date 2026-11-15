"""
Evaluate SegFormer-B2 on CFP test set:
1. Row-normalized confusion matrix heatmap (results/figures/b2_confusion_matrix.png)
2. 8-sample prediction overlay grid (results/figures/b2_test_predictions_grid.png)
3. Full-image evaluation metrics (results/b2_test_full_metrics.json)
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


def load_b2_model(checkpoint_path="models/segformer_b2_best.pt", device="cpu"):
    cfg = AutoConfig.from_pretrained("nvidia/segformer-b2-finetuned-ade-512-512")
    cfg.num_labels = NUM_CLASSES
    cfg.id2label = {i: name for i, name in CLASS_NAMES.items()}
    cfg.label2id = {name: i for i, name in CLASS_NAMES.items()}
    model = SegformerForSemanticSegmentation(cfg)

    ckpt = torch.load(checkpoint_path, map_location=device)
    # Handle Hugging Face transformers naming changes between versions if present
    if set(ckpt.keys()) != set(model.state_dict().keys()):
        remapped = {}
        for k, v in ckpt.items():
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
            remapped[k_new] = v
        ckpt = remapped

    model.load_state_dict(ckpt)
    model.to(device)
    model.eval()
    return model


def predict_full_image(model, img: Image.Image, device="cpu"):
    w, h = img.size
    # Pad to multiple of 32 for SegFormer patch grid
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

    # Crop padding back to original image size
    return pred[:h, :w]


def mask_to_color(mask):
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for cid, color in CLASS_COLORS_RGB.items():
        rgb[mask == cid] = color
    return rgb


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading SegFormer-B2 on {device}...")
    model = load_b2_model("models/segformer_b2_best.pt", device=device)

    test_img_paths = sorted((DATA_PROCESSED / "test" / "images").glob("*.png"))
    test_msk_paths = sorted((DATA_PROCESSED / "test" / "masks").glob("*.png"))
    print(f"Running inference on {len(test_img_paths)} test images...")

    confusion_matrix = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    predictions = {}

    for i, (ip, mp) in enumerate(zip(test_img_paths, test_msk_paths)):
        stem = ip.stem
        img = Image.open(ip).convert("RGB")
        gt = np.array(Image.open(mp))

        pred = predict_full_image(model, img, device=device)
        predictions[stem] = pred

        # Accumulate confusion matrix (gt as row, pred as column)
        valid = (gt >= 0) & (gt < NUM_CLASSES)
        np.add.at(confusion_matrix, (gt[valid], pred[valid]), 1)

        if (i + 1) % 15 == 0 or (i + 1) == len(test_img_paths):
            print(f"  Processed {i+1}/{len(test_img_paths)}")

    # ── 1. Row-Normalized Confusion Matrix ──────────────────────────────────
    row_sums = confusion_matrix.sum(axis=1, keepdims=True)
    norm_cm = np.divide(confusion_matrix.astype(np.float64), row_sums, out=np.zeros_like(confusion_matrix, dtype=np.float64), where=row_sums > 0)

    class_labels = [CLASS_NAMES[c] for c in range(NUM_CLASSES)]

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(norm_cm, cmap="Blues", vmin=0, vmax=1.0)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(np.arange(NUM_CLASSES))
    ax.set_yticks(np.arange(NUM_CLASSES))
    ax.set_xticklabels(class_labels, rotation=45, ha="right", fontsize=10)
    ax.set_yticklabels(class_labels, fontsize=10)
    ax.set_xlabel("Predicted Class", fontsize=11, labelpad=10)
    ax.set_ylabel("Ground Truth Class", fontsize=11, labelpad=10)
    ax.set_title("SegFormer-B2 Row-Normalized Confusion Matrix (Recall %)", fontsize=13, pad=15)

    # Annotate percentages
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

    # ── 2. Overlay Grid on 8 Test Images ────────────────────────────────────
    random.seed(42)
    sample_indices = random.sample(range(len(test_img_paths)), 8)

    fig, axes = plt.subplots(8, 3, figsize=(15, 3.2 * 8))

    for row_idx, idx in enumerate(sample_indices):
        ip = test_img_paths[idx]
        stem = ip.stem
        img = np.array(Image.open(ip))
        gt = np.array(Image.open(test_msk_paths[idx]))
        pred = predictions[stem]

        axes[row_idx, 0].imshow(img)
        axes[row_idx, 0].set_title(f"{stem} (Image)", fontsize=9)
        axes[row_idx, 0].axis("off")

        axes[row_idx, 1].imshow(img)
        axes[row_idx, 1].imshow(mask_to_color(gt), alpha=0.55)
        axes[row_idx, 1].set_title("Ground Truth", fontsize=9)
        axes[row_idx, 1].axis("off")

        axes[row_idx, 2].imshow(img)
        axes[row_idx, 2].imshow(mask_to_color(pred), alpha=0.55)
        axes[row_idx, 2].set_title("SegFormer-B2 Prediction", fontsize=9)
        axes[row_idx, 2].axis("off")

    patches = [mpatches.Patch(color=np.array(CLASS_COLORS_RGB[c]) / 255.0, label=f"{c}: {CLASS_NAMES[c]}") for c in range(NUM_CLASSES)]
    fig.legend(handles=patches, loc="lower center", ncol=5, fontsize=10, bbox_to_anchor=(0.5, 0.005))
    plt.suptitle("SegFormer-B2 Test Set Predictions vs Ground Truth", fontsize=14, y=0.995)
    plt.tight_layout(rect=[0, 0.03, 1, 0.99])

    grid_path = FIGURES_DIR / "b2_test_predictions_grid.png"
    plt.savefig(grid_path, dpi=120)
    plt.close()
    print(f"✓ 8-image overlay grid saved to {grid_path}")


if __name__ == "__main__":
    main()


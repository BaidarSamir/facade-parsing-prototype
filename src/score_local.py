"""
Score B2 predictions against COCO 1.0 polygon annotations on Bolzano photos.

For each annotated image (skipping images with 0 annotations):
  - Load photo with EXIF rotation, verify size matches JSON width/height
  - Rasterize COCO polygons to binary masks (window → class 2, door → class 5)
  - Resize photo (longest side 1024) and masks (nearest), run B2 inference
  - Report binary IoU, precision, recall for window and door per image and pooled

Outputs:
  results/bolzano_scored.json   — per-image and pooled metrics
  results/figures/bolzano_scored_grid.png — side-by-side comparison
"""

import json, re, sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageOps
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
IMAGES_DIR = ROOT / "data" / "bolzano_photos"
ANNO_FILE = ROOT / "annotations" / "instances_default.json"
WEIGHTS = ROOT / "models" / "segformer_b2_best.pt"
OUT_JSON = ROOT / "results" / "bolzano_scored.json"
OUT_FIG = ROOT / "results" / "figures" / "bolzano_scored_grid.png"
OUT_FIG.parent.mkdir(parents=True, exist_ok=True)

LONG_SIDE = 1024
NAMES = ["background", "building", "window", "sky", "roof",
         "door", "tree", "people", "car", "sign"]
PALETTE = np.array([(0,0,0),(70,70,70),(250,170,30),(70,130,180),(0,60,100),
                    (153,153,153),(107,142,35),(255,0,0),(0,0,142),(220,220,0)],
                   dtype=np.uint8)
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

# Map COCO category NAME → model class id
CATNAME_TO_CLS = {"window": 2, "door": 5}

# ── Model loading (identical to predict_overlays.py) ─────────────────────────
from transformers import SegformerForSemanticSegmentation


def convert_keys(sd):
    rules = [
        (r"^segformer\.stages\.(\d+)\.patch_embeddings\.", r"segformer.encoder.patch_embeddings.\1."),
        (r"^segformer\.stages\.(\d+)\.layer_norm\.", r"segformer.encoder.layer_norm.\1."),
        (r"^segformer\.stages\.(\d+)\.blocks\.(\d+)\.", r"segformer.encoder.block.\1.\2."),
        (r"\.layernorm_before\.", ".layer_norm_1."),
        (r"\.layernorm_after\.", ".layer_norm_2."),
        (r"\.attention\.q_proj\.", ".attention.self.query."),
        (r"\.attention\.k_proj\.", ".attention.self.key."),
        (r"\.attention\.v_proj\.", ".attention.self.value."),
        (r"\.attention\.o_proj\.", ".attention.output.dense."),
        (r"\.attention\.sequence_reduction\.sequence_reduction\.", ".attention.self.sr."),
        (r"\.attention\.sequence_reduction\.layer_norm\.", ".attention.self.layer_norm."),
        (r"\.mlp\.fc1\.", ".mlp.dense1."),
        (r"\.mlp\.fc2\.", ".mlp.dense2."),
        (r"^decode_head\.linear_projections\.", "decode_head.linear_c."),
    ]
    out = {}
    for k, v in sd.items():
        for pat, rep in rules:
            k = re.sub(pat, rep, k)
        out[k] = v
    return out


def load_model(weights):
    m = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/segformer-b2-finetuned-ade-512-512",
        num_labels=10, ignore_mismatched_sizes=True)
    sd = torch.load(weights, map_location="cpu")
    try:
        m.load_state_dict(sd)
    except RuntimeError:
        m.load_state_dict(convert_keys(sd))
        print("Loaded after renaming checkpoint keys.")
    return m.eval()


# ── Image loading (identical to predict_overlays.py) ─────────────────────────

def load_image(path):
    img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    return img


def resize_long(img, long_side=LONG_SIDE):
    w, h = img.size
    s = long_side / max(w, h)
    if abs(s - 1) > 1e-6:
        img = img.resize((round(w * s), round(h * s)), Image.LANCZOS)
    return img


@torch.no_grad()
def predict(model, img):
    x = torch.from_numpy(np.array(img)).permute(2, 0, 1)[None].float() / 255.0
    x = (x - MEAN) / STD
    h, w = x.shape[-2:]
    ph, pw = (32 - h % 32) % 32, (32 - w % 32) % 32
    x = F.pad(x, (0, pw, 0, ph), mode="reflect")
    lg = model(pixel_values=x).logits
    lg = F.interpolate(lg, size=x.shape[-2:], mode="bilinear",
                       align_corners=False)[..., :h, :w]
    return lg.argmax(1)[0].numpy().astype(np.uint8)


# ── COCO polygon → binary mask rasterization ─────────────────────────────────

def rasterize_polygons(annotations, width, height, cat_name_map, catname_to_cls):
    """
    Rasterize COCO polygon annotations into per-class binary masks.

    Parameters
    ----------
    annotations : list of COCO annotation dicts (all for one image)
    width, height : image dimensions from COCO images entry
    cat_name_map : dict  COCO category id → category name string
    catname_to_cls : dict  category name → model class id

    Returns
    -------
    masks : dict  class_id → (H, W) bool array
    """
    masks = {}
    for cls_id in catname_to_cls.values():
        masks[cls_id] = np.zeros((height, width), dtype=bool)

    for ann in annotations:
        cat_name = cat_name_map.get(ann["category_id"])
        if cat_name is None or cat_name not in catname_to_cls:
            continue
        cls_id = catname_to_cls[cat_name]
        for seg in ann["segmentation"]:
            # seg is a flat list [x1,y1,x2,y2,...] — convert to list of (x,y) tuples
            poly = list(zip(seg[0::2], seg[1::2]))
            tmp = Image.new("L", (width, height), 0)
            ImageDraw.Draw(tmp).polygon(poly, fill=1)
            masks[cls_id] |= (np.array(tmp) > 0)

    return masks


# ── Binary metrics ───────────────────────────────────────────────────────────

def binary_metrics(gt_mask, pred_mask):
    """IoU, precision, recall between two bool arrays."""
    tp = int((gt_mask & pred_mask).sum())
    fp = int((~gt_mask & pred_mask).sum())
    fn = int((gt_mask & ~pred_mask).sum())
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else float("nan")
    prec = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    rec = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    return {"iou": round(iou, 4), "precision": round(prec, 4),
            "recall": round(rec, 4), "tp": tp, "fp": fp, "fn": fn}


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    # Load COCO annotations
    with open(ANNO_FILE) as f:
        coco = json.load(f)

    cat_name_map = {c["id"]: c["name"] for c in coco["categories"]}
    print(f"COCO categories: {cat_name_map}")

    # Group annotations by image_id
    anns_by_img = {}
    for ann in coco["annotations"]:
        anns_by_img.setdefault(ann["image_id"], []).append(ann)

    # Filter images: skip those with 0 annotations
    images_to_score = []
    for img_info in coco["images"]:
        n_anns = len(anns_by_img.get(img_info["id"], []))
        if n_anns == 0:
            print(f"  SKIP {img_info['file_name']}: 0 annotations")
            continue
        images_to_score.append(img_info)

    if not images_to_score:
        print("No images with annotations to score.")
        sys.exit(1)

    print(f"\nScoring {len(images_to_score)} images "
          f"(classes: {list(CATNAME_TO_CLS.keys())})\n")

    # Load model
    model = load_model(WEIGHTS)

    per_image_results = []
    grid_rows = []  # (name, photo, gt_vis, pred_vis)

    # Accumulators for pooled metrics
    pooled = {}
    for cls_name, cls_id in CATNAME_TO_CLS.items():
        pooled[cls_name] = {"tp": 0, "fp": 0, "fn": 0}

    for img_info in images_to_score:
        fname = img_info["file_name"]
        json_w, json_h = img_info["width"], img_info["height"]
        img_path = IMAGES_DIR / fname

        if not img_path.exists():
            print(f"  [WARN] {fname} not found in {IMAGES_DIR} — skipping")
            continue

        # Load with EXIF rotation
        photo = load_image(img_path)
        actual_w, actual_h = photo.size

        # Size check
        if actual_w != json_w or actual_h != json_h:
            print(f"  [ERROR] {fname}: photo is {actual_w}x{actual_h} "
                  f"but JSON says {json_w}x{json_h}. "
                  f"EXIF-transposed size does not match annotation dimensions.")
            sys.exit(1)

        # Rasterize GT polygons at original resolution
        gt_masks = rasterize_polygons(
            anns_by_img[img_info["id"]], json_w, json_h,
            cat_name_map, CATNAME_TO_CLS)

        # Resize photo and GT masks
        photo_resized = resize_long(photo, LONG_SIDE)
        rw, rh = photo_resized.size

        gt_masks_resized = {}
        for cls_id, m in gt_masks.items():
            m_pil = Image.fromarray(m.astype(np.uint8) * 255)
            m_pil = m_pil.resize((rw, rh), Image.NEAREST)
            gt_masks_resized[cls_id] = np.array(m_pil) > 0

        # B2 prediction
        pred_full = predict(model, photo_resized)

        # Score each class
        img_result = {"file": fname, "size_original": f"{json_w}x{json_h}",
                      "size_resized": f"{rw}x{rh}"}
        for cls_name, cls_id in CATNAME_TO_CLS.items():
            gt_bin = gt_masks_resized[cls_id]
            pred_bin = (pred_full == cls_id)
            m = binary_metrics(gt_bin, pred_bin)
            img_result[cls_name] = m
            pooled[cls_name]["tp"] += m["tp"]
            pooled[cls_name]["fp"] += m["fp"]
            pooled[cls_name]["fn"] += m["fn"]
            print(f"  {fname} | {cls_name:6s} | "
                  f"IoU={m['iou']:.3f}  prec={m['precision']:.3f}  "
                  f"rec={m['recall']:.3f}")

        per_image_results.append(img_result)

        # Build a GT visualisation mask (0 = unlabelled, cls_id where annotated)
        gt_vis = np.zeros((rh, rw), dtype=np.uint8)
        for cls_id, m in gt_masks_resized.items():
            gt_vis[m] = cls_id

        grid_rows.append((fname, np.array(photo_resized), gt_vis, pred_full))

    # ── Pooled metrics ───────────────────────────────────────────────────────
    pooled_results = {}
    print(f"\n{'='*60}")
    print("POOLED (across all scored images)")
    print(f"{'='*60}")
    for cls_name in CATNAME_TO_CLS:
        p = pooled[cls_name]
        tp, fp, fn = p["tp"], p["fp"], p["fn"]
        iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else float("nan")
        prec = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
        rec = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        pooled_results[cls_name] = {
            "iou": round(iou, 4), "precision": round(prec, 4),
            "recall": round(rec, 4), "tp": tp, "fp": fp, "fn": fn
        }
        print(f"  {cls_name:6s} | IoU={iou:.3f}  prec={prec:.3f}  rec={rec:.3f}  "
              f"(tp={tp:,}  fp={fp:,}  fn={fn:,})")

    # ── Save JSON ────────────────────────────────────────────────────────────
    output = {
        "description": "B2 vs COCO polygon annotations on Bolzano photos",
        "classes_evaluated": list(CATNAME_TO_CLS.keys()),
        "category_name_to_model_class": CATNAME_TO_CLS,
        "n_images_scored": len(per_image_results),
        "pooled": pooled_results,
        "per_image": per_image_results,
    }
    with open(OUT_JSON, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n✓ Metrics saved → {OUT_JSON}")

    # ── Grid figure ──────────────────────────────────────────────────────────
    n = len(grid_rows)
    fig, axes = plt.subplots(n, 3, figsize=(14, n * 3.5), squeeze=False)
    col_titles = ["Photo", "My Annotations (GT)", "B2 Prediction"]

    for r, (name, photo, gt_vis, pred) in enumerate(grid_rows):
        axes[r][0].imshow(photo)
        axes[r][0].set_title(name, fontsize=9)

        # GT: show only annotated classes, rest as dark grey
        gt_rgb = np.full((*gt_vis.shape, 3), 40, dtype=np.uint8)
        for cls_id in CATNAME_TO_CLS.values():
            gt_rgb[gt_vis == cls_id] = PALETTE[cls_id]
        axes[r][1].imshow(gt_rgb)
        axes[r][1].set_title(col_titles[1], fontsize=9)

        axes[r][2].imshow(PALETTE[pred])
        axes[r][2].set_title(col_titles[2], fontsize=9)

        for a in axes[r]:
            a.axis("off")

    # Legend for the evaluated classes
    handles = [Patch(color=PALETTE[CATNAME_TO_CLS[n]] / 255.0, label=n)
               for n in CATNAME_TO_CLS]
    handles.append(Patch(facecolor=(0.16, 0.16, 0.16), label="unlabelled"))
    fig.legend(handles=handles, loc="lower center", ncol=len(handles),
               fontsize=9, frameon=False)

    fig.suptitle(f"B2 vs Manual Annotations — Bolzano ({n} images)", fontsize=12)
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    fig.savefig(OUT_FIG, dpi=120)
    plt.close(fig)
    print(f"✓ Figure saved → {OUT_FIG}")


if __name__ == "__main__":
    main()

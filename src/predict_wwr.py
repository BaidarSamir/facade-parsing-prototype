"""
Window-to-Wall Ratio (WWR) Prediction Script
Runs SegFormer-B2 inference on the 61 test images and exports
per-image WWR predictions alongside ground-truth WWR.
"""

import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation

from config import DATA_PROCESSED, MODELS_DIR, RESULTS_DIR

CKPT = MODELS_DIR / "segformer_b2_best.pt"
TEST_DIR = DATA_PROCESSED / "test"
IMG_DIR = TEST_DIR / "images"
MSK_DIR = TEST_DIR / "masks"
OUT_JSON = RESULTS_DIR / "wwr_predictions.json"
NUM_CLASSES = 10
MIN_GROSS_PX = 1000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def remap_ckpt(state):
    import re
    new = {}
    for k, v in state.items():
        k2 = re.sub(r"segformer\.stages\.(\d+)\.", lambda m: f"segformer.encoder.block.{m.group(1)}.", k)
        k2 = k2.replace("segformer.patch_embeddings.", "segformer.encoder.patch_embeddings.")
        k2 = k2.replace("segformer.layer_norms.", "segformer.encoder.layer_norms.")
        k2 = k2.replace(".linear_projections.", ".linear_c.")
        new[k2] = v
    return new

def load_model():
    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/segformer-b2-finetuned-ade-512-512",
        num_labels=NUM_CLASSES,
        ignore_mismatched_sizes=True,
    )
    raw = torch.load(CKPT, map_location="cpu")
    state = raw.get("model_state_dict", raw)
    model.load_state_dict(remap_ckpt(state), strict=False)
    return model.to(device).eval()

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

def wwr_from_mask(m):
    win = int((m == 2).sum())
    bld = int((m == 1).sum())
    door = int((m == 5).sum())
    gross = win + bld + door
    return (win / gross if gross >= MIN_GROSS_PX else None), gross

def main():
    model = load_model()
    img_paths = sorted(IMG_DIR.glob("*.png"))
    mask_paths = sorted(MSK_DIR.glob("*.png"))
    assert len(img_paths) == len(mask_paths) == 61, f"Expected 61 test images, got {len(img_paths)}"

    records = []
    for img_p, msk_p in zip(img_paths, mask_paths):
        gt = np.array(Image.open(msk_p))
        gt = gt[..., 0] if gt.ndim == 3 else gt
        gt_wwr, gt_gross = wwr_from_mask(gt)

        img = np.array(Image.open(img_p).convert("RGB"))
        t = (torch.from_numpy(img).float().permute(2, 0, 1) / 255.0 - MEAN) / STD
        inp = t.unsqueeze(0)
        h, w = inp.shape[-2:]
        ph, pw = (32 - h % 32) % 32, (32 - w % 32) % 32
        x = F.pad(inp, (0, pw, 0, ph), mode="reflect").to(device)

        with torch.no_grad():
            logits = model(pixel_values=x).logits
        up = F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)[..., :h, :w]
        pred = up.argmax(1).squeeze(0).cpu().numpy()
        pred_wwr, pred_gross = wwr_from_mask(pred)

        records.append({
            "stem": img_p.stem,
            "gt_wwr": gt_wwr,
            "pred_wwr": pred_wwr,
            "gt_gross_px": gt_gross,
            "pred_gross_px": pred_gross,
        })

    with open(OUT_JSON, "w") as f:
        json.dump(records, f, indent=2)
    print(f"Saved {len(records)} records -> {OUT_JSON}")

if __name__ == "__main__":
    main()

"""SegFormer-B2 overlays and grid figures.

Bolzano photos:
  python src/predict_overlays.py --images-dir data/bolzano_photos --out results/bolzano
CFP test set with ground truth:
  python src/predict_overlays.py --images-dir data/cfp_processed/test/images --masks-dir data/cfp_processed/test/masks --out results/cfp_grid --n 8 --wwr-json results/wwr_predictions.json
"""
import argparse, csv, json, re
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from transformers import SegformerForSemanticSegmentation

NAMES = ["background", "building", "window", "sky", "roof", "door", "tree", "people", "car", "sign"]
PALETTE = np.array([(0, 0, 0), (70, 70, 70), (250, 170, 30), (70, 130, 180), (0, 60, 100),
                    (153, 153, 153), (107, 142, 35), (255, 0, 0), (0, 0, 142), (220, 220, 0)], dtype=np.uint8)
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
LONG_SIDE = 1024
EXTS = {".jpg", ".jpeg", ".png"}
ROWS_PER_PAGE = 6


def convert_keys(sd):
    """Checkpoint saved with newer transformers key names; map them to the older names."""
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
        "nvidia/segformer-b2-finetuned-ade-512-512", num_labels=10, ignore_mismatched_sizes=True)
    sd = torch.load(weights, map_location="cpu")
    try:
        m.load_state_dict(sd)
    except RuntimeError:
        m.load_state_dict(convert_keys(sd))  # strict: raises an error if any key still mismatches
        print("Loaded after renaming checkpoint keys.")
    return m.eval()


def load_image(path):
    img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    w, h = img.size
    s = LONG_SIDE / max(w, h)
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
    lg = F.interpolate(lg, size=x.shape[-2:], mode="bilinear", align_corners=False)[..., :h, :w]
    return lg.argmax(1)[0].numpy().astype(np.uint8)


def wwr(mask, min_px=1000):
    win, bld, door = int((mask == 2).sum()), int((mask == 1).sum()), int((mask == 5).sum())
    gross = win + bld + door
    return win / gross if gross >= min_px else None


def blend(img, pred):
    return (0.5 * np.array(img) + 0.5 * PALETTE[pred]).astype(np.uint8)


def save_grid(rows, out_path, has_gt):
    cols = 3 if has_gt else 2
    fig, axes = plt.subplots(len(rows), cols, figsize=(cols * 4.4, len(rows) * 3.3), squeeze=False)
    for r, (name, img, gt, pred) in enumerate(rows):
        axes[r][0].imshow(img)
        axes[r][0].set_title(name, fontsize=8)
        c = 1
        if has_gt:
            g = wwr(gt)
            axes[r][1].imshow(PALETTE[gt])
            axes[r][1].set_title("ground truth" + (f" (WWR {g * 100:.0f}%)" if g is not None else ""), fontsize=8)
            c = 2
        p = wwr(pred)
        axes[r][c].imshow(PALETTE[pred])
        axes[r][c].set_title("B2 prediction" + (f" (WWR {p * 100:.0f}%)" if p is not None else ""), fontsize=8)
        for a in axes[r]:
            a.axis("off")
    handles = [Patch(color=PALETTE[i] / 255.0, label=NAMES[i]) for i in range(10)]
    fig.legend(handles=handles, loc="lower center", ncol=10, fontsize=7, frameon=False)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images-dir", required=True)
    ap.add_argument("--masks-dir", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--weights", default="models/segformer_b2_best.pt")
    ap.add_argument("--n", type=int, default=None, help="use n evenly spaced images")
    ap.add_argument("--wwr-json", default=None, help="add the 2 worst WWR error images (needs --masks-dir)")
    args = ap.parse_args()

    out = Path(args.out)
    (out / "overlays").mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in Path(args.images_dir).iterdir() if p.is_file())
    paths = [p for p in files if p.suffix.lower() in EXTS]
    skipped = [p.name for p in files if p.suffix.lower() not in EXTS]
    if skipped:
        print("Skipped (unsupported type):", skipped[:10])
    if not paths:
        raise SystemExit("No images found")

    chosen = paths
    if args.n and args.n < len(paths):
        idx = sorted(set(np.linspace(0, len(paths) - 1, args.n).round().astype(int).tolist()))
        chosen = [paths[i] for i in idx]
    if args.wwr_json and args.masks_dir:
        recs = [r for r in json.load(open(args.wwr_json)) if r["gt_wwr"] is not None and r["pred_wwr"] is not None]
        recs.sort(key=lambda r: abs(r["pred_wwr"] - r["gt_wwr"]), reverse=True)
        by_stem = {p.stem: p for p in paths}
        for r in recs[:2]:
            p = by_stem.get(r["stem"])
            if p is not None and p not in chosen:
                chosen.append(p)

    model = load_model(args.weights)
    rows, table = [], []
    for k, p in enumerate(chosen, 1):
        img = load_image(p)
        pred = predict(model, img)
        gt = None
        if args.masks_dir:
            mp = Path(args.masks_dir) / (p.stem + ".png")
            if mp.exists():
                g = Image.open(mp)
                if g.size != img.size:
                    g = g.resize(img.size, Image.NEAREST)
                gt = np.array(g)
                gt = gt[..., 0] if gt.ndim == 3 else gt
        Image.fromarray(blend(img, pred)).save(out / "overlays" / f"{p.stem}_overlay.png")
        pct = np.bincount(pred.ravel(), minlength=10) / pred.size * 100
        w = wwr(pred)
        table.append([p.name, "" if w is None else f"{w * 100:.1f}"] + [f"{v:.1f}" for v in pct])
        rows.append((p.name, img, gt, pred))
        print(f"[{k}/{len(chosen)}] {p.name}  WWR={'n/a' if w is None else f'{w * 100:.0f}%'}  "
              + " ".join(f"{NAMES[i]}={pct[i]:.0f}%" for i in range(10) if pct[i] >= 1))

    has_gt = args.masks_dir is not None and all(r[2] is not None for r in rows)
    if args.masks_dir and not has_gt:
        print("Some ground-truth masks were missing, grid will show photo and prediction only.")
    for i in range(0, len(rows), ROWS_PER_PAGE):
        save_grid(rows[i:i + ROWS_PER_PAGE], out / f"grid_{i // ROWS_PER_PAGE + 1:02d}.png", has_gt)
    with open(out / "summary.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["file", "pred_wwr_percent"] + [f"{n}_percent" for n in NAMES])
        wr.writerows(table)
    print("Saved to", out)


if __name__ == "__main__":
    main()
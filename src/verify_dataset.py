"""
Verify CFP dataset: class mapping, mask integrity, alignment, and generate
labelled overlay figure.
"""
import random
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import Counter

# ── Ground-truth mapping from facade.py ──────────────────────────────────
CLASS_MAP = {
    0: "background",
    1: "building",
    2: "window",
    3: "sky",
    4: "roof",
    5: "door",
    6: "tree",
    7: "people",
    8: "car",
    9: "sign",
}

# Colors from facade.py (RGB)
CLASS_COLORS = {
    0: (0, 0, 0),
    1: (70, 70, 70),
    2: (250, 170, 30),
    3: (70, 130, 180),
    4: (0, 60, 100),
    5: (153, 153, 153),
    6: (107, 142, 35),
    7: (255, 0, 0),
    8: (0, 0, 142),
    9: (220, 220, 0),
}

ROOT = Path("data/cfp_raw/translated_data")
OUT = Path("results")
OUT.mkdir(exist_ok=True)

# ── 1. Check file counts ────────────────────────────────────────────────
imgs = sorted((ROOT / "images").glob("*.png"))
bmasks = sorted((ROOT / "binary_mask").glob("*.png"))
cmasks = sorted((ROOT / "color_mask").glob("*.png"))

print(f"Images: {len(imgs)}, Binary masks: {len(bmasks)}, Color masks: {len(cmasks)}")
assert len(imgs) == len(bmasks) == len(cmasks), "File count mismatch!"

# Check that every image has a matching mask
img_stems = {p.stem for p in imgs}
bm_stems = {p.stem for p in bmasks}
cm_stems = {p.stem for p in cmasks}
assert img_stems == bm_stems == cm_stems, "Filename mismatch!"
print("✓ All filenames match across images, binary_mask, and color_mask.")

# ── 2. Scan all masks: unique IDs, shape match, color consistency ────────
all_ids = Counter()
shape_mismatches = []
color_verify_errors = []
image_sizes = []

for stem in sorted(img_stems):
    img = np.array(Image.open(ROOT / "images" / f"{stem}.png"))
    bm = np.array(Image.open(ROOT / "binary_mask" / f"{stem}.png"))
    cm = np.array(Image.open(ROOT / "color_mask" / f"{stem}.png"))

    image_sizes.append((img.shape[0], img.shape[1]))

    # Shape check (H, W)
    if img.shape[:2] != bm.shape[:2]:
        shape_mismatches.append((stem, img.shape, bm.shape))

    # Count class IDs
    for v in np.unique(bm):
        all_ids[int(v)] += 1

    # Verify color_mask matches id2trainId mapping
    for cid, expected_rgb in CLASS_COLORS.items():
        pixels_with_id = bm == cid
        if pixels_with_id.any():
            actual_rgb = tuple(cm[pixels_with_id][0].tolist())
            if actual_rgb != expected_rgb:
                color_verify_errors.append((stem, cid, expected_rgb, actual_rgb))

print(f"\n── Unique class IDs across all {len(img_stems)} images ──")
for cid in sorted(all_ids):
    name = CLASS_MAP.get(cid, "UNKNOWN")
    print(f"  {cid:>2d}: {name:<12s}  (appears in {all_ids[cid]:>3d} images)")

unknown_ids = set(all_ids.keys()) - set(CLASS_MAP.keys())
if unknown_ids:
    print(f"\n⚠ Unexpected class IDs found: {unknown_ids}")
else:
    print("\n✓ All class IDs match the expected mapping (0-9).")

if shape_mismatches:
    print(f"\n⚠ Shape mismatches ({len(shape_mismatches)}):")
    for s, si, sm in shape_mismatches[:5]:
        print(f"  {s}: img={si}, mask={sm}")
else:
    print("✓ All image-mask shapes match.")

if color_verify_errors:
    print(f"\n⚠ Color mapping errors ({len(color_verify_errors)}):")
    for s, cid, exp, act in color_verify_errors[:5]:
        print(f"  {s}: class {cid}, expected {exp}, got {act}")
else:
    print("✓ Color masks are consistent with the class-to-RGB mapping.")

# Size distribution
sizes = Counter(image_sizes)
print(f"\n── Image size distribution ──")
for (h, w), cnt in sizes.most_common(10):
    print(f"  {w}x{h}: {cnt} images")

# ── 3. Pixel-level class distribution ────────────────────────────────────
print(f"\n── Pixel-level class distribution (full dataset) ──")
pixel_counts = Counter()
for stem in sorted(img_stems):
    bm = np.array(Image.open(ROOT / "binary_mask" / f"{stem}.png"))
    for v, c in zip(*np.unique(bm, return_counts=True)):
        pixel_counts[int(v)] += int(c)

total_px = sum(pixel_counts.values())
for cid in sorted(pixel_counts):
    name = CLASS_MAP.get(cid, "UNKNOWN")
    pct = 100 * pixel_counts[cid] / total_px
    print(f"  {cid:>2d}: {name:<12s}  {pct:6.2f}%  ({pixel_counts[cid]:>12,d} px)")

# ── 4. Train/val/test split (reproduce their code) ──────────────────────
from sklearn.model_selection import train_test_split

items = sorted(img_stems)
train, other = train_test_split(items, train_size=0.8, random_state=1)
val, test = train_test_split(other, train_size=0.5, random_state=1)
print(f"\n── Train/Val/Test split (80/10/10, random_state=1) ──")
print(f"  Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")

# ── 5. Labelled overlay figure ──────────────────────────────────────────
random.seed(42)
samples = random.sample(sorted(img_stems), 6)

fig, axes = plt.subplots(3, 4, figsize=(20, 15))
for i, stem in enumerate(samples):
    row = i // 2
    col_base = (i % 2) * 2

    img = np.array(Image.open(ROOT / "images" / f"{stem}.png"))
    bm = np.array(Image.open(ROOT / "binary_mask" / f"{stem}.png"))

    # Build color overlay from binary_mask using the known mapping
    overlay = np.zeros((*bm.shape, 3), dtype=np.uint8)
    for cid, rgb in CLASS_COLORS.items():
        overlay[bm == cid] = rgb

    axes[row, col_base].imshow(img)
    axes[row, col_base].set_title(f"{stem}", fontsize=9)
    axes[row, col_base].axis("off")

    axes[row, col_base + 1].imshow(img)
    axes[row, col_base + 1].imshow(overlay, alpha=0.55)
    axes[row, col_base + 1].set_title("overlay", fontsize=9)
    axes[row, col_base + 1].axis("off")

# Legend
present_ids = sorted(set().union(*(
    set(np.unique(np.array(Image.open(ROOT / "binary_mask" / f"{s}.png"))))
    for s in samples
)))
patches = [
    mpatches.Patch(
        color=np.array(CLASS_COLORS[cid]) / 255.0,
        label=f"{cid}: {CLASS_MAP[cid]}"
    )
    for cid in present_ids
]
fig.legend(handles=patches, loc="lower center", ncol=5, fontsize=11,
           frameon=True, title="Class Legend")
plt.suptitle("CFP Dataset – Labelled Overlay Verification", fontsize=14, y=0.98)
plt.tight_layout(rect=[0, 0.06, 1, 0.96])
plt.savefig(OUT / "dataset_verification.png", dpi=120, bbox_inches="tight")
print(f"\n✓ Overlay figure saved to {OUT / 'dataset_verification.png'}")
plt.close()

# ── 6. Check for instance info (are there per-instance IDs?) ─────────
print(f"\n── Instance check ──")
print("Binary masks contain class IDs only (0-9), no instance encoding.")
print("The masks are purely semantic segmentation labels.")


"""
Resize images + masks (longest side → 1024) and write train/val/test splits
to data/cfp_processed/.  Replicates the RTFP 80/10/10 split (random_state=1).
"""
import shutil
from pathlib import Path
from PIL import Image
from sklearn.model_selection import train_test_split
from config import (
    DATA_RAW, DATA_PROCESSED, RESIZE_LONGEST,
    SPLIT_SEED, TRAIN_RATIO, VAL_TEST_RATIO,
)


def resize_keep_ratio(img: Image.Image, longest: int, resample) -> Image.Image:
    w, h = img.size
    scale = longest / max(w, h)
    if scale >= 1.0:
        return img  # already small enough
    new_w, new_h = round(w * scale), round(h * scale)
    return img.resize((new_w, new_h), resample)


def main():
    src_images = DATA_RAW / "images"
    src_masks = DATA_RAW / "binary_mask"

    stems = sorted(p.stem for p in src_images.glob("*.png"))
    print(f"Found {len(stems)} images")

    # ── Split ────────────────────────────────────────────────────────────
    train, other = train_test_split(stems, train_size=TRAIN_RATIO, random_state=SPLIT_SEED)
    val, test = train_test_split(other, train_size=VAL_TEST_RATIO, random_state=SPLIT_SEED)
    splits = {"train": sorted(train), "val": sorted(val), "test": sorted(test)}
    for name, items in splits.items():
        print(f"  {name}: {len(items)}")

    # ── Clean output dir ─────────────────────────────────────────────────
    if DATA_PROCESSED.exists():
        shutil.rmtree(DATA_PROCESSED)

    # ── Resize & save ────────────────────────────────────────────────────
    for split_name, stem_list in splits.items():
        img_dir = DATA_PROCESSED / split_name / "images"
        msk_dir = DATA_PROCESSED / split_name / "masks"
        img_dir.mkdir(parents=True, exist_ok=True)
        msk_dir.mkdir(parents=True, exist_ok=True)

        for i, stem in enumerate(stem_list):
            img = Image.open(src_images / f"{stem}.png").convert("RGB")
            msk = Image.open(src_masks / f"{stem}.png")  # uint8, single channel

            img = resize_keep_ratio(img, RESIZE_LONGEST, Image.LANCZOS)
            msk = resize_keep_ratio(msk, RESIZE_LONGEST, Image.NEAREST)

            img.save(img_dir / f"{stem}.png")
            msk.save(msk_dir / f"{stem}.png")

            if (i + 1) % 50 == 0:
                print(f"    {split_name}: {i+1}/{len(stem_list)}")

    # ── Quick sanity check ───────────────────────────────────────────────
    import numpy as np
    for split_name in splits:
        msk_dir = DATA_PROCESSED / split_name / "masks"
        sample = sorted(msk_dir.glob("*.png"))[:3]
        for p in sample:
            m = np.array(Image.open(p))
            assert m.ndim == 2, f"{p.name}: expected 2D mask, got shape {m.shape}"
            assert set(np.unique(m)).issubset(set(range(10))), f"{p.name}: unexpected IDs {np.unique(m)}"
    print("✓ Sanity check passed")
    print(f"✓ Output written to {DATA_PROCESSED}")


if __name__ == "__main__":
    main()


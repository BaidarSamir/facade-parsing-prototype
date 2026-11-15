"""
PyTorch Dataset for SegFormer fine-tuning on CFP.
Works identically on local CPU and Kaggle GPU.
"""
import random
import numpy as np
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T


class FacadeDataset(Dataset):
    """
    Reads image/mask PNGs.
    Can be initialized with:
      - base root + split: FacadeDataset("data/cfp_processed", split="train")
      - direct split folder: FacadeDataset("/kaggle/input/.../train", split="train")
    Masks are single-channel uint8 with class IDs 0-9.
    """

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        crop_size: int = 512,
        augment: bool = True,
    ):
        p = Path(root)
        if (p / split).exists():
            self.root = p / split
        elif (p / "images").exists():
            self.root = p
        else:
            self.root = p / split

        self.crop_size = crop_size
        self.augment = augment and (split == "train")

        self.image_paths = sorted((self.root / "images").glob("*.png"))
        self.mask_paths = sorted((self.root / "masks").glob("*.png"))
        assert len(self.image_paths) == len(self.mask_paths), \
            f"Mismatch in {self.root}: {len(self.image_paths)} images vs {len(self.mask_paths)} masks"
        assert len(self.image_paths) > 0, f"No images found in {self.root / 'images'}"

        self.normalize = T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        msk = Image.open(self.mask_paths[idx])  # single-channel uint8

        if self.augment:
            img, msk = self._augment(img, msk)
        else:
            img, msk = self._center_crop_or_pad(img, msk)

        img = T.ToTensor()(img)                                # [3, H, W] float32 0-1
        img = self.normalize(img)                              # ImageNet norm
        msk = torch.from_numpy(np.array(msk, dtype=np.int64))  # [H, W] long

        return {"pixel_values": img, "labels": msk}

    def _augment(self, img: Image.Image, msk: Image.Image):
        img, msk = self._random_crop(img, msk, self.crop_size)

        if random.random() > 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            msk = msk.transpose(Image.FLIP_LEFT_RIGHT)

        img = T.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25)(img)
        return img, msk

    def _random_crop(self, img, msk, size):
        w, h = img.size
        if w < size or h < size:
            img, msk = self._pad(img, msk, size)
            w, h = img.size
        x = random.randint(0, w - size)
        y = random.randint(0, h - size)
        img = img.crop((x, y, x + size, y + size))
        msk = msk.crop((x, y, x + size, y + size))
        return img, msk

    def _center_crop_or_pad(self, img, msk):
        w, h = img.size
        size = self.crop_size
        if w < size or h < size:
            img, msk = self._pad(img, msk, size)
            w, h = img.size
        x = (w - size) // 2
        y = (h - size) // 2
        img = img.crop((x, y, x + size, y + size))
        msk = msk.crop((x, y, x + size, y + size))
        return img, msk

    def _pad(self, img, msk, size):
        w, h = img.size
        pw = max(0, size - w)
        ph = max(0, size - h)
        if pw > 0 or ph > 0:
            img = Image.fromarray(np.pad(
                np.array(img),
                ((0, ph), (0, pw), (0, 0)),
                mode="reflect",
            ))
            msk = Image.fromarray(np.pad(
                np.array(msk),
                ((0, ph), (0, pw)),
                mode="constant", constant_values=0,
            ))
        return img, msk


def compute_class_weights(root_or_split: str | Path, split: str = "train", num_classes: int = 10):
    """
    Inverse-frequency weights, capped at 10x median.
    Accepts base root or direct train folder.
    """
    p = Path(root_or_split)
    if (p / split / "masks").exists():
        mask_dir = p / split / "masks"
    elif (p / "masks").exists():
        mask_dir = p / "masks"
    else:
        mask_dir = p / split / "masks"

    counts = np.zeros(num_classes, dtype=np.float64)
    for mp in sorted(mask_dir.glob("*.png")):
        m = np.array(Image.open(mp))
        for c in range(num_classes):
            counts[c] += (m == c).sum()

    freq = counts / (counts.sum() + 1e-8)
    weights = 1.0 / (freq + 1e-8)
    cap = 10.0 * np.median(weights)
    weights = np.minimum(weights, cap)
    weights /= weights.sum() / num_classes  # normalize mean to 1.0
    return torch.tensor(weights, dtype=torch.float32)

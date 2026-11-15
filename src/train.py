"""
Fine-tune SegFormer (B0 or B2) on CFP dataset.
Can be run via CLI locally or on Kaggle GPU.
"""
import argparse
import json
import math
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import SegformerForSemanticSegmentation, get_cosine_schedule_with_warmup

from config import CLASS_NAMES, NUM_CLASSES
from dataset import FacadeDataset, compute_class_weights


MODEL_CHECKPOINTS = {
    "b0": "nvidia/segformer-b0-finetuned-ade-512-512",
    "b2": "nvidia/segformer-b2-finetuned-ade-512-512",
}


def compute_dataset_iou(total_intersection, total_union):
    ious = {}
    valid_ious_10 = []
    valid_ious_9 = []
    for c in range(NUM_CLASSES):
        name = CLASS_NAMES[c]
        u = total_union[c]
        if u > 0:
            iou = total_intersection[c] / u
            ious[name] = float(iou)
            valid_ious_10.append(iou)
            if c > 0:
                valid_ious_9.append(iou)
        else:
            ious[name] = None

    miou_10 = float(np.mean(valid_ious_10)) if valid_ious_10 else 0.0
    miou_9 = float(np.mean(valid_ious_9)) if valid_ious_9 else 0.0
    return ious, miou_10, miou_9


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_intersection = np.zeros(NUM_CLASSES, dtype=np.float64)
    total_union = np.zeros(NUM_CLASSES, dtype=np.float64)
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    for batch in loader:
        images = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(pixel_values=images)
        logits = F.interpolate(
            outputs.logits,
            size=labels.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        loss = criterion(logits, labels)
        total_loss += loss.item() * images.size(0)

        preds = logits.argmax(dim=1).cpu().numpy()
        gts = labels.cpu().numpy()

        for c in range(NUM_CLASSES):
            pred_c = preds == c
            gt_c = gts == c
            total_intersection[c] += np.logical_and(pred_c, gt_c).sum()
            total_union[c] += np.logical_or(pred_c, gt_c).sum()

    avg_loss = total_loss / len(loader.dataset)
    ious, miou_10, miou_9 = compute_dataset_iou(total_intersection, total_union)
    return avg_loss, ious, miou_10, miou_9


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device} | Model: SegFormer-{args.model_type.upper()}")

    output_dir = Path(args.output_dir) / f"segformer_{args.model_type.lower()}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Datasets & Loaders ──
    train_dataset = FacadeDataset(args.train_dir, split="train", crop_size=args.crop_size, augment=True)
    val_dataset = FacadeDataset(args.val_dir, split="val", crop_size=args.crop_size, augment=False)
    test_dataset = FacadeDataset(args.test_dir, split="test", crop_size=args.crop_size, augment=False)

    print(f"Dataset sizes: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # ── Loss with Class Weights ──
    print("Computing class weights from training set...")
    class_weights = compute_class_weights(args.train_dir, split="train", num_classes=NUM_CLASSES)
    print("Class weights (0-9):", [round(w, 2) for w in class_weights.tolist()])
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    # ── Model ──
    pretrained_id = MODEL_CHECKPOINTS[args.model_type.lower()]
    id2label = {i: name for i, name in CLASS_NAMES.items()}
    label2id = {name: i for i, name in CLASS_NAMES.items()}

    model = SegformerForSemanticSegmentation.from_pretrained(
        pretrained_id,
        num_labels=NUM_CLASSES,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )
    model.to(device)

    # ── Optimizer & Scheduler ──
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(0.05 * total_steps)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    start_epoch = 1
    best_miou = 0.0
    history = []

    # ── Resume Checkpoint if available ──
    latest_ckpt = output_dir / "checkpoint_latest.pt"
    if args.resume and latest_ckpt.exists():
        print(f"Resuming from {latest_ckpt}...")
        checkpoint = torch.load(latest_ckpt, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_miou = checkpoint["best_miou"]
        history = checkpoint.get("history", [])
        print(f"Resumed at epoch {start_epoch} (best val mIoU without bg so far: {best_miou*100:.2f}%)")

    # ── Training Loop ──
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        train_loss = 0.0
        t0 = time.time()

        for step, batch in enumerate(train_loader):
            images = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                outputs = model(pixel_values=images)
                logits = F.interpolate(
                    outputs.logits,
                    size=labels.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            train_loss += loss.item() * images.size(0)

        train_loss /= len(train_loader.dataset)
        elapsed = time.time() - t0

        # ── Validation ──
        val_loss, val_ious, val_miou_10, val_miou_9 = evaluate(model, val_loader, device)

        print(
            f"Epoch [{epoch:02d}/{args.epochs:02d}] ({elapsed:.1f}s) | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Val mIoU (10cls): {val_miou_10*100:.2f}% | Val mIoU (no bg): {val_miou_9*100:.2f}%"
        )

        epoch_record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_miou_10": val_miou_10,
            "val_miou_9_no_bg": val_miou_9,
            "val_ious": val_ious,
        }
        history.append(epoch_record)

        # ── Regular Checkpointing (for session limits) ──
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_miou": best_miou,
                "history": history,
            },
            latest_ckpt,
        )

        # ── Best Model Checkpoint (based on mIoU without bg) ──
        if val_miou_9 > best_miou:
            best_miou = val_miou_9
            torch.save(model.state_dict(), output_dir / "best_model.pt")
            print(f"  --> Best model saved! (Val mIoU no bg: {best_miou*100:.2f}%)")

        with open(output_dir / "train_history.json", "w") as f:
            json.dump(history, f, indent=2)

    # ── Final Test Evaluation with Best Model ──
    print("\n" + "=" * 50)
    print("Loading best model for Test Evaluation...")
    best_weights = output_dir / "best_model.pt"
    if best_weights.exists():
        model.load_state_dict(torch.load(best_weights, map_location=device))

    test_loss, test_ious, test_miou_10, test_miou_9 = evaluate(model, test_loader, device)
    print(f"TEST RESULTS (SegFormer-{args.model_type.upper()}):")
    print(f"  Test Loss: {test_loss:.4f}")
    print(f"  mIoU (10 classes): {test_miou_10*100:.2f}%")
    print(f"  mIoU (9 classes, no bg): {test_miou_9*100:.2f}%")
    print("  Per-class IoU:")
    for name, iou_val in test_ious.items():
        if iou_val is not None:
            print(f"    {name:<12s}: {iou_val*100:5.2f}%")
        else:
            print(f"    {name:<12s}: N/A")

    final_metrics = {
        "model_type": args.model_type.upper(),
        "test_loss": test_loss,
        "test_miou_10": test_miou_10,
        "test_miou_9_no_bg": test_miou_9,
        "per_class_iou": test_ious,
    }
    with open(output_dir / "test_metrics.json", "w") as f:
        json.dump(final_metrics, f, indent=2)
    print(f"Artifacts saved in {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Train SegFormer on CFP dataset")
    parser.add_argument("--model_type", type=str, default="b0", choices=["b0", "b2"])
    parser.add_argument("--train_dir", type=str, default="data/cfp_processed/train")
    parser.add_argument("--val_dir", type=str, default="data/cfp_processed/val")
    parser.add_argument("--test_dir", type=str, default="data/cfp_processed/test")
    parser.add_argument("--output_dir", type=str, default="results/training")
    parser.add_argument("--crop_size", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--lr", type=float, default=6e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--resume", action="store_true", default=True)
    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()


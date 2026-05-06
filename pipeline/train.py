#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .preprocessing import get_datasets

RESULTS_DIR = Path(__file__).parent.parent / "results"
BIOCLIP_FREEZE_EPOCHS = 5
EARLY_STOP_PATIENCE = 5


def build_model(model_name, backbone, num_classes):
    if model_name == "baseline":
        from models.baseline import BaselineModel
        return BaselineModel(num_classes, backbone)
    if model_name == "early_fusion":
        from models.early_fusion import EarlyFusionModel
        return EarlyFusionModel(num_classes, backbone)
    if model_name == "late_fusion":
        from models.late_fusion import LateFusionModel
        return LateFusionModel(num_classes, backbone)
    if model_name == "location_only":
        from models.location_only import LocationOnlyModel
        return LocationOnlyModel(num_classes)
    raise ValueError(f"Unknown model: {model_name}")


def make_optimizer(model, model_name):
    if model_name == "location_only":
        return torch.optim.Adam(model.parameters(), lr=1e-3)
    backbone_ids = {id(p) for p in model.backbone.parameters()}
    other_params = [p for p in model.parameters() if id(p) not in backbone_ids]
    return torch.optim.Adam([
        {"params": list(model.backbone.parameters()), "lr": 1e-5},
        {"params": other_params, "lr": 1e-3},
    ])


def out_dir(model_name, backbone):
    if model_name == "location_only":
        return RESULTS_DIR / "location_only"
    return RESULTS_DIR / f"{model_name}_{backbone}"


def compute_class_weights(train_ds, num_classes, device):
    counts = torch.zeros(num_classes)
    for sample in train_ds.samples:
        counts[train_ds.label_to_idx[sample[-1]]] += 1
    counts = counts.clamp(min=1)
    weights = 1.0 / counts
    return (weights / weights.sum() * num_classes).to(device)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for img, loc, label in loader:
        img, loc, label = img.to(device), loc.to(device), label.to(device)
        optimizer.zero_grad()
        loss = criterion(model(img, loc), label)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(label)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct = 0.0, 0
    for img, loc, label in loader:
        img, loc, label = img.to(device), loc.to(device), label.to(device)
        logits = model(img, loc)
        total_loss += criterion(logits, label).item() * len(label)
        correct += (logits.argmax(1) == label).sum().item()
    n = len(loader.dataset)
    return total_loss / n, correct / n


def save_checkpoint(path, model, optimizer, scheduler, epoch, best_val_loss):
    torch.save({
        "model":         model.state_dict(),
        "optimizer":     optimizer.state_dict(),
        "scheduler":     scheduler.state_dict(),
        "epoch":         epoch,
        "best_val_loss": best_val_loss,
    }, path)


def load_checkpoint(path, model, optimizer, scheduler):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    return ckpt["epoch"], ckpt["best_val_loss"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",      required=True,
                        choices=["baseline", "early_fusion", "late_fusion", "location_only"])
    parser.add_argument("--backbone",   default="resnet50", choices=["resnet50", "bioclip"])
    parser.add_argument("--epochs",     type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--resume",     action="store_true",
                        help="Resume training from last.pt if it exists")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    result_dir = out_dir(args.model, args.backbone)
    result_dir.mkdir(parents=True, exist_ok=True)

    print("Loading datasets...")
    train_ds, val_ds, _, num_classes, label_to_idx = get_datasets()
    print(f"Classes: {num_classes} | train: {len(train_ds)} | val: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=4, pin_memory=True)

    print("Building model...")
    model     = build_model(args.model, args.backbone, num_classes).to(device)
    optimizer = make_optimizer(model, args.model)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss(weight=compute_class_weights(train_ds, num_classes, device))

    start_epoch    = 0
    best_val_loss  = float("inf")
    patience_count = 0
    history_path   = result_dir / "history.json"
    history: list  = []

    if args.resume:
        last_pt = result_dir / "last.pt"
        if last_pt.exists():
            start_epoch, best_val_loss = load_checkpoint(last_pt, model, optimizer, scheduler)
            start_epoch += 1
            print(f"Resumed from epoch {start_epoch} (best_val_loss={best_val_loss:.4f})")
            if history_path.exists():
                with open(history_path) as f:
                    history = json.load(f)
        else:
            print("--resume set but no last.pt found; starting from scratch")

    uses_bioclip = (args.model != "location_only" and args.backbone == "bioclip")
    if uses_bioclip and start_epoch < BIOCLIP_FREEZE_EPOCHS:
        for p in model.backbone.parameters():
            p.requires_grad_(False)
        print(f"BioCLIP backbone frozen for first {BIOCLIP_FREEZE_EPOCHS} epochs")

    for epoch in range(start_epoch, args.epochs):
        if uses_bioclip and epoch == BIOCLIP_FREEZE_EPOCHS:
            for p in model.backbone.parameters():
                p.requires_grad_(True)
            print(f"Epoch {epoch}: BioCLIP backbone unfrozen")

        train_loss            = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc     = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        print(f"Epoch {epoch:3d} | train_loss={train_loss:.4f} | "
              f"val_loss={val_loss:.4f} | val_acc={val_acc:.4f}")

        history.append({"epoch": epoch, "train_loss": train_loss,
                         "val_loss": val_loss, "val_acc": val_acc})
        with open(history_path, "w") as f:
            json.dump(history, f)

        save_checkpoint(result_dir / "last.pt", model, optimizer, scheduler, epoch, best_val_loss)

        if val_loss < best_val_loss:
            best_val_loss  = val_loss
            patience_count = 0
            save_checkpoint(result_dir / "best.pt", model, optimizer, scheduler, epoch, best_val_loss)
            print(f"  -> best model saved (val_loss={best_val_loss:.4f})")
        else:
            patience_count += 1
            if patience_count >= EARLY_STOP_PATIENCE:
                print(f"Early stopping at epoch {epoch} (no improvement for {EARLY_STOP_PATIENCE} epochs)")
                break

    with open(result_dir / "label_to_idx.json", "w") as f:
        json.dump({k: int(v) for k, v in label_to_idx.items()}, f, indent=2)

    print(f"\nDone. Results in {result_dir}/")


if __name__ == "__main__":
    main()

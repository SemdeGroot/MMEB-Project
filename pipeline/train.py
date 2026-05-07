#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .preprocessing import get_datasets

RESULTS_DIR          = Path(__file__).parent.parent / "results"
BIOCLIP_FREEZE_EPOCHS = 5
EARLY_STOP_PATIENCE   = 5
LAMBDA                = 0.05   # contrastive loss weight for coordination model

# Models that have no image backbone (metadata/location only)
METADATA_ONLY_MODELS = {"location_only", "metadata_only"}

# Models that return (logits, z_img, z_meta) and use contrastive loss
COORDINATION_MODELS = {"coordination"}


# ---- Loss ----

class FocalLoss(nn.Module):
    """Focal loss that down-weights easy examples and focuses on hard ones."""

    def __init__(self, weight=None, gamma=2.0):
        super().__init__()
        self.weight = weight
        self.gamma  = gamma

    def forward(self, logits, targets):
        ce   = F.cross_entropy(logits, targets, weight=self.weight, reduction="none")
        pt   = torch.exp(-ce)
        loss = ((1 - pt) ** self.gamma) * ce
        return loss.mean()


# ---- Model factory ----

def build_model(model_name, backbone, num_classes, metadata_dim):
    if model_name == "baseline":
        from models.baseline import BaselineModel
        return BaselineModel(num_classes, backbone)
    if model_name == "location_only":
        from models.location_only import LocationOnlyModel
        return LocationOnlyModel(num_classes, metadata_dim)
    if model_name == "metadata_only":
        from models.metadata_only import MetadataOnlyModel
        return MetadataOnlyModel(num_classes, metadata_dim)
    if model_name == "early_fusion":
        from models.early_fusion import EarlyFusionModel
        return EarlyFusionModel(num_classes, backbone, metadata_dim)
    if model_name == "late_fusion":
        from models.late_fusion import LateFusionModel
        return LateFusionModel(num_classes, backbone, metadata_dim)
    if model_name == "gated_fusion":
        from models.gated_fusion import GatedFusionModel
        return GatedFusionModel(num_classes, backbone, metadata_dim)
    if model_name == "concat_fusion":
        from models.concat_fusion import ConcatFusionModel
        return ConcatFusionModel(num_classes, metadata_dim, backbone)
    if model_name == "transformer_fusion":
        from models.transformer_fusion import TransformerFusionModel
        return TransformerFusionModel(num_classes, metadata_dim, backbone)
    if model_name == "coordination":
        from models.coordination import CoordinationModel
        return CoordinationModel(num_classes, metadata_dim, backbone)
    raise ValueError(f"Unknown model: {model_name}")


def make_optimizer(model, model_name):
    if model_name in METADATA_ONLY_MODELS:
        return torch.optim.Adam(model.parameters(), lr=1e-3)
    backbone_ids = {id(p) for p in model.backbone.parameters()}
    other_params = [p for p in model.parameters() if id(p) not in backbone_ids]
    return torch.optim.Adam([
        {"params": list(model.backbone.parameters()), "lr": 1e-5},
        {"params": other_params, "lr": 1e-3},
    ])


def out_dir(model_name, backbone):
    if model_name in METADATA_ONLY_MODELS:
        return RESULTS_DIR / model_name
    return RESULTS_DIR / f"{model_name}_{backbone}"


def compute_class_weights(train_ds, num_classes, device):
    # Sqrt-inverse frequency keeps rare classes upweighted without destabilising training.
    counts = torch.zeros(num_classes)
    for sample in train_ds.samples:
        counts[train_ds.label_to_idx[sample[-1]]] += 1
    counts  = counts.clamp(min=1)
    weights = 1.0 / counts.sqrt()
    return (weights / weights.sum() * num_classes).to(device)


def _progress(tag, i, n, extra=""):
    filled = int(40 * (i + 1) / n)
    bar = "#" * filled + "-" * (40 - filled)
    print(f"  {tag} [{bar}] {i+1}/{n}{extra}", flush=True)


# ---- Train / eval loops ----

def train_one_epoch(model, loader, criterion, contrastive_loss,
                    optimizer, device, scaler, is_coordination):
    model.train()
    total_loss = 0.0
    n          = len(loader)
    interval   = max(1, n // 100)
    use_amp    = scaler is not None

    for i, (img, meta, label) in enumerate(loader):
        img, meta, label = img.to(device), meta.to(device), label.to(device)
        optimizer.zero_grad()

        if use_amp:
            with torch.amp.autocast(device_type="cuda"):
                loss = _compute_loss(model, img, meta, label, criterion,
                                     contrastive_loss, is_coordination)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss = _compute_loss(model, img, meta, label, criterion,
                                 contrastive_loss, is_coordination)
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * len(label)
        if (i + 1) % interval == 0 or i + 1 == n:
            avg = total_loss / ((i + 1) * loader.batch_size)
            _progress("train", i, n, f"  loss={avg:.4f}")

    return total_loss / len(loader.dataset)


def _compute_loss(model, img, meta, label, criterion, contrastive_loss, is_coordination):
    out = model(img, meta)
    if isinstance(out, tuple):
        logits, z_img, z_meta = out
        loss = criterion(logits, label)
        if is_coordination:
            loss = loss + LAMBDA * contrastive_loss(z_img, z_meta)
    else:
        loss = criterion(out, label)
    return loss


@torch.no_grad()
def evaluate(model, loader, criterion, contrastive_loss, device, use_amp, is_coordination):
    model.eval()
    total_loss, correct = 0.0, 0
    n        = len(loader)
    interval = max(1, n // 100)

    for i, (img, meta, label) in enumerate(loader):
        img, meta, label = img.to(device), meta.to(device), label.to(device)
        if use_amp:
            with torch.amp.autocast(device_type="cuda"):
                loss, logits = _eval_step(model, img, meta, label, criterion,
                                          contrastive_loss, is_coordination)
        else:
            loss, logits = _eval_step(model, img, meta, label, criterion,
                                      contrastive_loss, is_coordination)

        total_loss += loss.item() * len(label)
        correct    += (logits.argmax(1) == label).sum().item()
        if (i + 1) % interval == 0 or i + 1 == n:
            _progress("val  ", i, n)

    n_samples = len(loader.dataset)
    return total_loss / n_samples, correct / n_samples


def _eval_step(model, img, meta, label, criterion, contrastive_loss, is_coordination):
    out = model(img, meta)
    if isinstance(out, tuple):
        logits, z_img, z_meta = out
        loss = criterion(logits, label)
        if is_coordination:
            loss = loss + LAMBDA * contrastive_loss(z_img, z_meta)
    else:
        logits = out
        loss   = criterion(logits, label)
    return loss, logits


# ---- Checkpointing ----

def save_checkpoint(path, model, optimizer, scheduler, epoch, best_val_loss):
    torch.save({
        "model":         model.state_dict(),
        "optimizer":     optimizer.state_dict(),
        "scheduler":     scheduler.state_dict(),
        "epoch":         epoch,
        "best_val_loss": best_val_loss,
    }, path)


def load_checkpoint(path, model, optimizer, scheduler):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    return ckpt["epoch"], ckpt["best_val_loss"]


# ---- Main ----

def main():
    all_models = [
        "baseline", "location_only", "metadata_only",
        "early_fusion", "late_fusion", "gated_fusion",
        "concat_fusion", "transformer_fusion", "coordination",
    ]
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",      required=True, choices=all_models)
    parser.add_argument("--backbone",   default="resnet50", choices=["resnet50", "bioclip"])
    parser.add_argument("--epochs",     type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--resume",     action="store_true",
                        help="Resume training from last.pt if it exists")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    result_dir = out_dir(args.model, args.backbone)
    result_dir.mkdir(parents=True, exist_ok=True)

    print("Loading datasets...")
    train_ds, val_ds, _, num_classes, label_to_idx, metadata_dim = get_datasets(
        backbone=args.backbone)
    print(f"Classes: {num_classes} | train: {len(train_ds)} | val: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=8, pin_memory=True,
                              persistent_workers=True, prefetch_factor=2)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=8, pin_memory=True,
                              persistent_workers=True, prefetch_factor=2)

    print("Building model...")
    model     = build_model(args.model, args.backbone, num_classes, metadata_dim).to(device)
    optimizer = make_optimizer(model, args.model)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    class_weights = compute_class_weights(train_ds, num_classes, device)
    criterion     = FocalLoss(weight=class_weights)

    is_coordination = args.model in COORDINATION_MODELS
    if is_coordination:
        from models.coordination import NTXentLoss
        contrastive_loss = NTXentLoss(temperature=0.5).to(device)
    else:
        contrastive_loss = None

    use_amp = device.type == "cuda"
    scaler  = torch.amp.GradScaler("cuda") if use_amp else None
    if use_amp:
        print("Mixed precision (autocast + GradScaler) enabled")

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

    uses_bioclip = (args.model not in METADATA_ONLY_MODELS and args.backbone == "bioclip")
    if uses_bioclip and start_epoch < BIOCLIP_FREEZE_EPOCHS:
        for p in model.backbone.parameters():
            p.requires_grad_(False)
        print(f"BioCLIP backbone frozen for first {BIOCLIP_FREEZE_EPOCHS} epochs")

    for epoch in range(start_epoch, args.epochs):
        if uses_bioclip and epoch == BIOCLIP_FREEZE_EPOCHS:
            for p in model.backbone.parameters():
                p.requires_grad_(True)
            print(f"Epoch {epoch}: BioCLIP backbone unfrozen")

        train_loss = train_one_epoch(
            model, train_loader, criterion, contrastive_loss,
            optimizer, device, scaler, is_coordination)
        val_loss, val_acc = evaluate(
            model, val_loader, criterion, contrastive_loss,
            device, use_amp, is_coordination)

        # Hold LR schedule while BioCLIP backbone is frozen
        if not (uses_bioclip and epoch < BIOCLIP_FREEZE_EPOCHS):
            scheduler.step()

        print(f"Epoch {epoch:3d} | train_loss={train_loss:.4f} | "
              f"val_loss={val_loss:.4f} | val_acc={val_acc:.4f}")

        history.append({"epoch": epoch, "train_loss": train_loss,
                        "val_loss": val_loss, "val_acc": val_acc})
        with open(history_path, "w") as f:
            json.dump(history, f)

        save_checkpoint(result_dir / "last.pt", model, optimizer, scheduler,
                        epoch, best_val_loss)

        if val_loss < best_val_loss:
            best_val_loss  = val_loss
            patience_count = 0
            save_checkpoint(result_dir / "best.pt", model, optimizer, scheduler,
                            epoch, best_val_loss)
            print(f"  -> best model saved (val_loss={best_val_loss:.4f})")
        else:
            patience_count += 1
            if patience_count >= EARLY_STOP_PATIENCE:
                print(f"Early stopping at epoch {epoch} "
                      f"(no improvement for {EARLY_STOP_PATIENCE} epochs)")
                break

    with open(result_dir / "label_to_idx.json", "w") as f:
        json.dump({k: int(v) for k, v in label_to_idx.items()}, f, indent=2)

    print(f"\nDone. Results in {result_dir}/")


if __name__ == "__main__":
    main()

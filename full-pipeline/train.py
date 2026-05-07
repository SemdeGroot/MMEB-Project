import os
import sys
import argparse
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

import config
from dataset import load_data, clean_data, preprocess_metadata, encode_labels, split_data, get_dataloaders, get_metadata_columns
from models.model_image import ImageOnlyModel
from models.model_metadata import MetadataOnlyModel
from models.model_concat_fusion import ConcatFusionModel
from models.model_gated_fusion import GatedFusionModel
from models.model_transformer_fusion import TransformerFusionModel
from models.model_coordination import TransformerFusionWithCoordination, NTXentLoss
from models.model_early_fusion import EarlyFusionModel
from models.model_cross_attention_transformer import CrossAttentionTransformerFusion


# ---- Directories ----
CHECKPOINT_DIR = "checkpoints"
RESULTS_DIR    = "results"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR,    exist_ok=True)

# ---- Lambda for contrastive loss ----
# Keep this small so classification remains the main objective.
LAMBDA = 0.05

# ---- Focal Loss for imbalanced classification ----
class FocalLoss(nn.Module):
    """
    Focal Loss (Lin et al., ICCV 2017).
    Addresses class imbalance by down-weighting easy examples and
    focusing training on hard, misclassified ones.

      FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)

    Args:
        alpha: overall scaling factor (default 1.0)
        gamma: focusing exponent - higher = more focus on hard examples
        reduction: 'mean' or 'sum'
    """
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha     = alpha
        self.gamma     = gamma
        self.reduction = reduction

    def forward(self, inputs, targets, class_weights=None):
        ce_loss    = F.cross_entropy(inputs, targets, reduction='none', weight=class_weights)
        p          = torch.exp(-ce_loss)                          # model confidence on correct class
        focal_loss = self.alpha * (1 - p) ** self.gamma * ce_loss  # upweight hard examples

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


# ---- Tee: write to both console and file ----
class Tee:
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()
    def flush(self):
        for f in self.files:
            f.flush()


def set_random_seed(seed):
    # Seed Python, NumPy, and PyTorch in one place for reproducible runs.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    deterministic = getattr(config, 'DETERMINISTIC_TRAINING', True)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = not deterministic


# ---- Model factory ----
def get_model(model_name, num_classes, metadata_dim):
    models = {
        'image_only':         ImageOnlyModel(num_classes),
        'metadata_only':      MetadataOnlyModel(num_classes, input_dim=metadata_dim),
        'concat_fusion':      ConcatFusionModel(num_classes, metadata_dim=metadata_dim),
        'gated_fusion':       GatedFusionModel(num_classes, metadata_dim=metadata_dim),
        'transformer_fusion': TransformerFusionModel(num_classes, metadata_dim=metadata_dim),
        'coordination':       TransformerFusionWithCoordination(num_classes, metadata_dim=metadata_dim),
        'early_fusion':       EarlyFusionModel(num_classes, metadata_dim=metadata_dim),
        'cross_attention_tf': CrossAttentionTransformerFusion(num_classes, metadata_dim=metadata_dim),
    }
    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}. Choose from: {list(models.keys())}")
    return models[model_name]


# ---- Training one epoch ----
def train_one_epoch(model, loader, optimizer, criterion, device, is_coordination=False):
    model.train()
    total_loss     = 0
    total_cls_loss = 0
    total_con_loss = 0
    correct        = 0
    total          = 0
    skipped        = 0

    contrastive_loss = NTXentLoss(temperature=0.5).to(device) if is_coordination else None

    for images, metadata, labels in loader:
        images   = images.to(device)
        metadata = metadata.to(device)
        labels   = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images, metadata)

        # Fusion models can optionally return embeddings alongside logits.
        if isinstance(outputs, tuple):
            logits, z_img, z_meta = outputs
            cls_loss = criterion(logits, labels)
            if is_coordination:
                # Coordination models optimise classification and cross-modal alignment together.
                con_loss = contrastive_loss(z_img, z_meta)
                loss     = cls_loss + LAMBDA * con_loss
                total_cls_loss += cls_loss.item()
                total_con_loss += con_loss.item()
            else:
                loss = cls_loss
            predicted = logits.argmax(dim=1)
        else:
            loss      = criterion(outputs, labels)
            predicted = outputs.argmax(dim=1)

        # Skip bad batches instead of stopping a long run completely.
        if not torch.isfinite(loss):
            skipped += 1
            optimizer.zero_grad()
            continue

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item()
        correct    += (predicted == labels).sum().item()
        total      += labels.size(0)

    if skipped > 0:
        sys.__stdout__.write(f"  [Warning] Skipped {skipped} batches due to NaN/Inf loss\n")
        sys.__stdout__.flush()

    avg_loss = total_loss / max(len(loader) - skipped, 1)
    accuracy = correct / max(total, 1) * 100

    if is_coordination:
        return avg_loss, total_cls_loss / max(len(loader) - skipped, 1), total_con_loss / max(len(loader) - skipped, 1), accuracy
    return avg_loss, accuracy


# ---- Evaluation ----
def evaluate(model, loader, criterion, device, is_coordination=False):
    model.eval()
    total_loss = 0
    correct    = 0
    total      = 0
    skipped    = 0

    contrastive_loss = NTXentLoss(temperature=0.5).to(device) if is_coordination else None

    with torch.no_grad():
        for images, metadata, labels in loader:
            images   = images.to(device)
            metadata = metadata.to(device)
            labels   = labels.to(device)

            outputs = model(images, metadata)

            if isinstance(outputs, tuple):
                logits, z_img, z_meta = outputs
                cls_loss = criterion(logits, labels)
                if is_coordination:
                    con_loss = contrastive_loss(z_img, z_meta)
                    loss     = cls_loss + LAMBDA * con_loss
                else:
                    loss = cls_loss
                predicted = logits.argmax(dim=1)
            else:
                loss      = criterion(outputs, labels)
                predicted = outputs.argmax(dim=1)

            # Skip NaN/Inf batches
            if not torch.isfinite(loss):
                skipped += 1
                continue

            total_loss += loss.item()
            correct    += (predicted == labels).sum().item()
            total      += labels.size(0)

    avg_loss = total_loss / max(len(loader) - skipped, 1)
    accuracy = correct / max(total, 1) * 100
    return avg_loss, accuracy


# ---- Full training loop ----
def train(model, train_loader, val_loader, model_name):
    device = config.DEVICE
    # Only coordination-style models add the contrastive term.
    is_coordination = model_name in ['coordination', 'cross_attention_tf']

    # Metadata-only is the hardest setting, so it gets slightly stronger focusing.
    if model_name == 'metadata_only':
        criterion = FocalLoss(alpha=1.0, gamma=2.5)
    else:
        criterion = FocalLoss(alpha=1.0, gamma=2.0)

    # Use conservative settings for the pretrained image baseline and a larger LR elsewhere.
    if model_name == 'metadata_only':
        base_lr      = 1e-4
        patience_val = 5
    elif model_name == 'image_only':
        base_lr      = config.LEARNING_RATE
        patience_val = 3
    else:
        base_lr      = 1e-4
        patience_val = 4

    optimizer = Adam(model.parameters(), lr=base_lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=patience_val, factor=0.5)
    checkpoint_path = os.path.join(CHECKPOINT_DIR, f"best_model_{model_name}.pth")

    best_val_acc = -1.0
    best_epoch   = 0

    print(f"\nModel:      {model_name}")
    print(f"Device:     {device}")
    print(f"Epochs:     {config.EPOCHS}")
    print(f"LR:         {base_lr}")
    print(f"Loss:       FocalLoss")
    print(f"Seed:       {config.RANDOM_SEED}")
    print(f"Checkpoint: {checkpoint_path}")

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total_params:,}")

    if is_coordination:
        print(f"\n{'Epoch':<8} {'Total Loss':<12} {'Cls Loss':<12} {'Con Loss':<12} {'Train Acc':<12} {'Val Loss':<12} {'Val Acc':<10}")
        print("-" * 80)
    else:
        print(f"\n{'Epoch':<8} {'Train Loss':<12} {'Train Acc':<12} {'Val Loss':<12} {'Val Acc':<10}")
        print("-" * 55)

    for epoch in range(1, config.EPOCHS + 1):

        if is_coordination:
            train_loss, cls_loss, con_loss, train_acc = train_one_epoch(
                model, train_loader, optimizer, criterion, device, is_coordination=True)
        else:
            train_loss, train_acc = train_one_epoch(
                model, train_loader, optimizer, criterion, device)

        val_loss, val_acc = evaluate(model, val_loader, criterion, device, is_coordination)
        scheduler.step(val_loss)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch   = epoch
            torch.save(model.state_dict(), checkpoint_path)

        if is_coordination:
            print(f"{epoch:<8} {train_loss:<12.4f} {cls_loss:<12.4f} {con_loss:<12.4f} {train_acc:<12.2f} {val_loss:<12.4f} {val_acc:<10.2f}")
        else:
            print(f"{epoch:<8} {train_loss:<12.4f} {train_acc:<12.2f} {val_loss:<12.4f} {val_acc:<10.2f}")

    print(f"\nBest val accuracy: {best_val_acc:.2f}% at epoch {best_epoch}")
    return model, checkpoint_path


# ---- Main ----
if __name__ == '__main__':

    # ---- Argument parser ----
    parser = argparse.ArgumentParser(description='Train a moth classification model')
    parser.add_argument('--model', type=str, required=True,
                        choices=['image_only', 'metadata_only', 'concat_fusion',
                                 'gated_fusion', 'transformer_fusion', 'coordination',
                                 'early_fusion', 'cross_attention_tf'],
                        help='Model to train')
    args       = parser.parse_args()
    model_name = args.model

    # ---- Redirect output to a per-model log file ----
    log_path = os.path.join(RESULTS_DIR, f"log_{model_name}.txt")
    log_file = open(log_path, 'w')
    sys.stdout = Tee(sys.__stdout__, log_file)

    print(f"Training log: {model_name}")
    print(f"{'='*60}")
    set_random_seed(config.RANDOM_SEED)

    # ---- Load and prepare data ----
    df = load_data()
    df = clean_data(df)
    df = preprocess_metadata(df)
    metadata_cols = get_metadata_columns(df)
    df, label_encoder = encode_labels(df)
    train_df, val_df, test_df = split_data(df)
    train_loader, val_loader, test_loader = get_dataloaders(train_df, val_df, test_df)
    num_classes = df['species'].nunique()
    metadata_dim = len(metadata_cols)

    # ---- Build and train model ----
    model = get_model(model_name, num_classes, metadata_dim).to(config.DEVICE)
    model, checkpoint_path = train(model, train_loader, val_loader, model_name)

    # ---- Final test evaluation on the best saved checkpoint ----
    is_coordination = model_name in ['coordination', 'cross_attention_tf']
    model.load_state_dict(torch.load(checkpoint_path, map_location=config.DEVICE))
    criterion = FocalLoss(alpha=1.0, gamma=2.0)
    test_loss, test_acc = evaluate(model, test_loader, criterion, config.DEVICE, is_coordination)

    print(f"\nTest Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.2f}%")

    # ---- Close log file ----
    log_file.close()
    sys.stdout = sys.__stdout__
    print(f"Log saved to {log_path}")

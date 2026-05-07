import os
import sys
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

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
RESULTS_DIR    = os.path.join("results", "advanced_evaluation")
FIGURES_DIR    = os.path.join(RESULTS_DIR, "figures")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

LAMBDA = 0.05


# ---- Tee: print to both console and file ----
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


# ---- Get predictions ----
def get_predictions(model, loader, device):
    model.eval()
    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for images, metadata, labels in loader:
            images   = images.to(device)
            metadata = metadata.to(device)

            outputs = model(images, metadata)
            # Handle all models that return (logits, z_img, z_meta)
            if isinstance(outputs, tuple):
                outputs = outputs[0]

            predicted = outputs.argmax(dim=1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    return np.array(all_labels), np.array(all_preds)


# ---- Per-species accuracy ----
def per_species_accuracy(labels, preds, label_encoder):
    species_names = label_encoder.classes_
    results = []

    for class_idx in np.unique(labels):
        mask    = labels == class_idx
        correct = (preds[mask] == labels[mask]).sum()
        total   = mask.sum()
        acc     = correct / total * 100
        results.append({'species': species_names[class_idx], 'accuracy': acc, 'n_samples': total})

    return pd.DataFrame(results).sort_values('accuracy', ascending=False)


# ---- Confusion matrix plot ----
def plot_confusion_matrix(labels, preds, label_encoder, title, save_path):
    top_classes     = np.bincount(labels).argsort()[-20:][::-1]
    mask            = np.isin(labels, top_classes)
    labels_filtered = labels[mask]
    preds_filtered  = preds[mask]
    species_names   = label_encoder.classes_[top_classes]
    cm              = confusion_matrix(labels_filtered, preds_filtered, labels=top_classes)

    plt.figure(figsize=(16, 14))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=species_names, yticklabels=species_names)
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.xticks(rotation=90, fontsize=7)
    plt.yticks(rotation=0,  fontsize=7)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


# ---- Per-species accuracy plot ----
def plot_per_species_accuracy(df, title, save_path, top_n=20):
    top    = df.head(top_n)
    bottom = df.tail(top_n)
    plot_df = pd.concat([top, bottom]).drop_duplicates()
    colors  = ['green' if acc >= 50 else 'red' for acc in plot_df['accuracy']]

    plt.figure(figsize=(12, 8))
    plt.barh(plot_df['species'], plot_df['accuracy'], color=colors)
    plt.xlabel('Accuracy (%)')
    plt.title(title)
    plt.axvline(x=50, color='black', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


# ---- Run evaluation for one model ----
def run_evaluation(model, test_loader, label_encoder, device, name):
    print(f"\n{'='*60}")
    print(f"ADVANCED EVALUATION: {name}")
    print(f"{'='*60}")

    labels, preds = get_predictions(model, test_loader, device)

    # ---- Per-species accuracy ----
    species_acc = per_species_accuracy(labels, preds, label_encoder)
    print("\nTop 10 easiest species:")
    print(species_acc.head(10).to_string(index=False))
    print("\nTop 10 hardest species:")
    print(species_acc.tail(10).to_string(index=False))

    # Classification report
    present       = np.unique(labels)
    present_names = label_encoder.classes_[present]
    report        = classification_report(labels, preds, labels=present,
                                          target_names=present_names, zero_division=0)
    print(f"\nClassification Report:\n{report}")

    # Save plots to results dir
    safe_name = name.lower().replace(' ', '_').replace('+', '_')
    plot_confusion_matrix(labels, preds, label_encoder,
                          title=f'Confusion Matrix - {name}',
                          save_path=os.path.join(FIGURES_DIR, f'confusion_matrix_{safe_name}.png'))
    plot_per_species_accuracy(species_acc,
                              title=f'Per-Species Accuracy - {name}',
                              save_path=os.path.join(FIGURES_DIR, f'per_species_accuracy_{safe_name}.png'))
    return labels, preds, species_acc


# ---- Main ----
if __name__ == '__main__':

    # ---- Redirect output ----
    log_path = os.path.join(RESULTS_DIR, "log_advanced_evaluation.txt")
    log_file = open(log_path, 'w')
    sys.stdout = Tee(sys.__stdout__, log_file)

    device = config.DEVICE

    # Load data
    df = load_data()
    df = clean_data(df)
    df = preprocess_metadata(df)
    metadata_cols = get_metadata_columns(df)
    df, label_encoder = encode_labels(df)
    train_df, val_df, test_df = split_data(df)
    _, _, test_loader = get_dataloaders(train_df, val_df, test_df)
    num_classes = df['species'].nunique()
    metadata_dim = len(metadata_cols)

    # All models to evaluate with their checkpoint names
    eval_models = {
        'Image Only':         ImageOnlyModel(num_classes),
        'Metadata Only':      MetadataOnlyModel(num_classes, input_dim=metadata_dim),
        'Early Fusion':       EarlyFusionModel(num_classes, metadata_dim=metadata_dim),
        'Concat Fusion':      ConcatFusionModel(num_classes, metadata_dim=metadata_dim),
        'Gated Fusion':       GatedFusionModel(num_classes, metadata_dim=metadata_dim),
        'Transformer Fusion': TransformerFusionModel(num_classes, metadata_dim=metadata_dim),
        'Coordination':       TransformerFusionWithCoordination(num_classes, metadata_dim=metadata_dim),
        'Cross-Attention TF': CrossAttentionTransformerFusion(num_classes, metadata_dim=metadata_dim),
    }
    ckpt_names = {
        'Image Only':         'best_model_image_only.pth',
        'Metadata Only':      'best_model_metadata_only.pth',
        'Early Fusion':       'best_model_early_fusion.pth',
        'Concat Fusion':      'best_model_concat_fusion.pth',
        'Gated Fusion':       'best_model_gated_fusion.pth',
        'Transformer Fusion': 'best_model_transformer_fusion.pth',
        'Coordination':       'best_model_coordination.pth',
        'Cross-Attention TF': 'best_model_cross_attention_tf.pth',
    }

    all_results = {}
    for name, model in eval_models.items():
        ckpt = os.path.join(CHECKPOINT_DIR, ckpt_names[name])
        if not os.path.exists(ckpt):
            print(f"\nSkipping {name} - checkpoint not found.")
            continue
        model = model.to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device))
        _, _, species_acc = run_evaluation(model, test_loader, label_encoder, device, name)
        all_results[name] = species_acc

    # ---- Comparison: which species benefit from Cross-Attention TF vs Transformer Fusion ----
    if 'Transformer Fusion' in all_results and 'Cross-Attention TF' in all_results:
        print(f"\n{'='*60}")
        print("COMPARISON: Transformer Fusion vs Cross-Attention TF")
        print(f"{'='*60}")

        merged = all_results['Transformer Fusion'].merge(
            all_results['Cross-Attention TF'], on='species', suffixes=('_tf', '_catf'))
        merged['improvement'] = merged['accuracy_catf'] - merged['accuracy_tf']
        merged = merged.sort_values('improvement', ascending=False)

        print("\nSpecies improved most with Cross-Attention TF:")
        print(merged.head(10)[['species', 'accuracy_tf', 'accuracy_catf', 'improvement']].to_string(index=False))
        print("\nSpecies degraded most with Cross-Attention TF:")
        print(merged.tail(10)[['species', 'accuracy_tf', 'accuracy_catf', 'improvement']].to_string(index=False))

    log_file.close()
    sys.stdout = sys.__stdout__
    print(f"Log saved to {log_path}")

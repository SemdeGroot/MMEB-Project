import os
import sys
from pathlib import Path
import torch
import torch.nn as nn

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
RESULTS_DIR    = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

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


# ---- Unified evaluation ----
def evaluate_standard(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct    = 0
    total      = 0

    with torch.no_grad():
        for images, metadata, labels in loader:
            images   = images.to(device)
            metadata = metadata.to(device)
            labels   = labels.to(device)
            outputs  = model(images, metadata)

            if isinstance(outputs, tuple):
                logits = outputs[0]
            else:
                logits = outputs

            loss = criterion(logits, labels)
            total_loss += loss.item()
            predicted = logits.argmax(dim=1)
            correct += (predicted == labels).sum().item()
            total   += labels.size(0)

    return total_loss / len(loader), correct / total * 100


# ---- Missing-modality test ----
def evaluate_missing_modality(model, loader, criterion, device, zero_out='metadata'):
    model.eval()
    correct = 0
    total   = 0

    with torch.no_grad():
        for images, metadata, labels in loader:
            images   = images.to(device)
            metadata = metadata.to(device)
            labels   = labels.to(device)

            if zero_out == 'metadata':
                metadata = torch.zeros_like(metadata)
            elif zero_out == 'image':
                images = torch.zeros_like(images)

            outputs = model(images, metadata)
            if isinstance(outputs, tuple):
                outputs = outputs[0]

            predicted = outputs.argmax(dim=1)
            correct += (predicted == labels).sum().item()
            total   += labels.size(0)

    return correct / total * 100


# ---- Main ----
if __name__ == '__main__':

    # ---- Redirect output ----
    log_path = os.path.join(RESULTS_DIR, "log_fission_analysis.txt")
    log_file = open(log_path, 'w')
    sys.stdout = Tee(sys.__stdout__, log_file)

    device    = config.DEVICE
    criterion = nn.CrossEntropyLoss()

    # Load the same cleaned test split used by training.
    df = load_data()
    df = clean_data(df)
    df = preprocess_metadata(df)
    metadata_cols = get_metadata_columns(df)
    df, label_encoder = encode_labels(df)
    train_df, val_df, test_df = split_data(df)
    _, _, test_loader = get_dataloaders(train_df, val_df, test_df)
    num_classes = df['species'].nunique()
    metadata_dim = len(metadata_cols)

    # ---- Load all available checkpoints ----
    models_config = {
        'Image-Only':           (ImageOnlyModel(num_classes).to(device),
                                 f'{CHECKPOINT_DIR}/best_model_image_only.pth'),
        'Metadata-Only':        (MetadataOnlyModel(num_classes, input_dim=metadata_dim).to(device),
                                 f'{CHECKPOINT_DIR}/best_model_metadata_only.pth'),
        'Early Fusion':         (EarlyFusionModel(num_classes, metadata_dim=metadata_dim).to(device),
                                 f'{CHECKPOINT_DIR}/best_model_early_fusion.pth'),
        'Concat Fusion':        (ConcatFusionModel(num_classes, metadata_dim=metadata_dim).to(device),
                                 f'{CHECKPOINT_DIR}/best_model_concat_fusion.pth'),
        'Gated Fusion':         (GatedFusionModel(num_classes, metadata_dim=metadata_dim).to(device),
                                 f'{CHECKPOINT_DIR}/best_model_gated_fusion.pth'),
        'Transformer Fusion':   (TransformerFusionModel(num_classes, metadata_dim=metadata_dim).to(device),
                                 f'{CHECKPOINT_DIR}/best_model_transformer_fusion.pth'),
        'Coordination':         (TransformerFusionWithCoordination(num_classes, metadata_dim=metadata_dim).to(device),
                                 f'{CHECKPOINT_DIR}/best_model_coordination.pth'),
        'Cross-Attention TF':   (CrossAttentionTransformerFusion(num_classes, metadata_dim=metadata_dim).to(device),
                                 f'{CHECKPOINT_DIR}/best_model_cross_attention_tf.pth'),
    }

    # ---- Step 1: Model comparison ----
    print("\n" + "="*60)
    print("STEP 1: MODEL COMPARISON")
    print("="*60)
    results = {}

    for name, (model, ckpt) in models_config.items():
        if not os.path.exists(ckpt):
            print(f"{name:<25} checkpoint not found, skipping.")
            continue
        model.load_state_dict(torch.load(ckpt, map_location=device))
        model.eval()
        _, acc = evaluate_standard(model, test_loader, criterion, device)
        results[name] = acc
        print(f"{name:<25} Test Accuracy: {acc:.2f}%")

    # ---- Step 2: Fission-style comparison ----
    print("\n" + "="*60)
    print("STEP 2: FISSION ANALYSIS")
    print("="*60)

    img_acc  = results.get('Image-Only',  0)
    meta_acc = results.get('Metadata-Only', 0)

    fusion_models = ['Early Fusion', 'Concat Fusion', 'Gated Fusion',
                     'Transformer Fusion', 'Coordination', 'Cross-Attention TF']
    fusion_results = {k: results[k] for k in fusion_models if k in results}

    best_fusion_name = max(fusion_results, key=fusion_results.get)
    best_fusion_acc  = fusion_results[best_fusion_name]

    print(f"\nUnique image info:      {img_acc:.2f}%")
    print(f"Unique metadata info:   {meta_acc:.2f}%")
    print(f"Best fusion model:      {best_fusion_name}")
    print(f"Best fusion accuracy:   {best_fusion_acc:.2f}%")

    synergy = best_fusion_acc - max(img_acc, meta_acc)
    print(f"\nSynergy (fusion - best unimodal): {synergy:+.2f}%")
    if synergy > 0:
        print("-> Fusion captures SYNERGISTIC information")
    else:
        print("-> No clear synergy detected")

    # Compare stronger attention-based fusion against the simpler baselines.
    simple_fusion_acc = max(
        results.get('Concat Fusion', 0),
        results.get('Gated Fusion', 0),
        results.get('Early Fusion', 0),
    )
    adv_acc = max(
        results.get('Coordination', 0),
        results.get('Cross-Attention TF', 0),
    )
    adv_benefit = adv_acc - simple_fusion_acc
    print(f"Advanced fusion benefit:          {adv_benefit:+.2f}%")
    if adv_benefit > 0:
        print("-> Advanced fusion (attention/contrastive) IMPROVES over simple fusion")
    else:
        print("-> Advanced fusion did not help over simple fusion")

    # ---- Step 3: Missing-modality test ----
    print("\n" + "="*60)
    print("STEP 3: MISSING MODALITY TEST")
    print("="*60)

    best_model, best_ckpt = models_config[best_fusion_name]
    best_model.load_state_dict(torch.load(best_ckpt, map_location=device))

    acc_no_meta  = evaluate_missing_modality(best_model, test_loader, criterion, device, zero_out='metadata')
    acc_no_image = evaluate_missing_modality(best_model, test_loader, criterion, device, zero_out='image')
    acc_full     = results[best_fusion_name]

    print(f"\nBest fusion model: {best_fusion_name}")
    print(f"Full modalities:   {acc_full:.2f}%")
    print(f"Missing metadata:  {acc_no_meta:.2f}%  (drop: {acc_full - acc_no_meta:+.2f}%)")
    print(f"Missing image:     {acc_no_image:.2f}%  (drop: {acc_full - acc_no_image:+.2f}%)")

    # ---- Step 4: Final summary ----
    print("\n" + "="*60)
    print("FINAL SUMMARY TABLE")
    print("="*60)
    print(f"\n{'Model':<25} {'Test Accuracy':>15}")
    print("-" * 42)
    for name, acc in results.items():
        print(f"{name:<25} {acc:>14.2f}%")

    log_file.close()
    sys.stdout = sys.__stdout__
    print(f"Log saved to {log_path}")

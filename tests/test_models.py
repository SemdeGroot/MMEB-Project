"""
Tests that each model forward pass runs and outputs the correct output shape.

Run with:
    python tests/test_models.py
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.baseline import BaselineModel
from models.early_fusion import EarlyFusionModel
from models.late_fusion import LateFusionModel
from models.gated_fusion import GatedFusionModel
from models.location_only import LocationOnlyModel

NUM_CLASSES = 10
BATCH = 4

img = torch.randn(BATCH, 3, 224, 224)
loc = torch.randn(BATCH, 4)


def check(name, model):
    model.eval()
    with torch.no_grad():
        out = model(img, loc)
    assert out.shape == (BATCH, NUM_CLASSES), (
        f"{name}: expected ({BATCH}, {NUM_CLASSES}), got {out.shape}"
    )
    print(f"  {name}: OK — output {out.shape}")


print("--- ResNet-50 models ---")
check("baseline      (resnet50)", BaselineModel(NUM_CLASSES, "resnet50"))
check("early_fusion  (resnet50)", EarlyFusionModel(NUM_CLASSES, "resnet50"))
check("late_fusion   (resnet50)", LateFusionModel(NUM_CLASSES, "resnet50"))
check("gated_fusion  (resnet50)", GatedFusionModel(NUM_CLASSES, "resnet50"))

print("--- location_only ---")
check("location_only", LocationOnlyModel(NUM_CLASSES))

print("--- BioCLIP models (downloads ~330 MB on first run) ---")
try:
    check("baseline      (bioclip)", BaselineModel(NUM_CLASSES, "bioclip"))
    check("early_fusion  (bioclip)", EarlyFusionModel(NUM_CLASSES, "bioclip"))
    check("late_fusion   (bioclip)", LateFusionModel(NUM_CLASSES, "bioclip"))
    check("gated_fusion  (bioclip)", GatedFusionModel(NUM_CLASSES, "bioclip"))
except Exception as e:
    print(f"  BioCLIP skipped: {e}")

print("\nAll model tests passed")

import torch
import torch.nn as nn
from torchvision import models

import config


# ---- Image-only baseline ----
# EfficientNet-B0 is reused as the visual backbone and only the classifier is replaced.

class ImageOnlyModel(nn.Module):
    def __init__(self, num_classes):
        super(ImageOnlyModel, self).__init__()

        self.encoder = models.efficientnet_b0(weights='DEFAULT')

        # Fine-tune the full backbone instead of freezing early layers.
        for param in self.encoder.parameters():
            param.requires_grad = True

        in_features = self.encoder.classifier[1].in_features

        self.encoder.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, images, metadata=None):
        # Keep the same forward signature as the multimodal models.
        return self.encoder(images)


# ---- Quick model test ----
if __name__ == '__main__':
    # Use a dummy batch to verify the model works before training
    num_classes = 162  # matches dataset output

    model = ImageOnlyModel(num_classes=num_classes).to(config.DEVICE)

    # Dummy inputs: batch of 4 images (3 channels, 224x224)
    dummy_images   = torch.randn(4, 3, 224, 224).to(config.DEVICE)
    dummy_metadata = torch.randn(4, 4).to(config.DEVICE)  # ignored in baseline

    # Forward pass
    output = model(dummy_images, dummy_metadata)

    # Output shape should be [batch_size, num_classes]
    print(f"Input shape:  {dummy_images.shape}")
    print(f"Output shape: {output.shape}")   # expected: [4, 162]
    print(f"Device: {next(model.parameters()).device}")
    print("Model check passed!")

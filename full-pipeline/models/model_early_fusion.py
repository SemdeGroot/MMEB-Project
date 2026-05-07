import torch
import torch.nn as nn
from torchvision import models

import config


# ---- Early fusion ----
# Image features and metadata are combined before the final shared MLP.

class EarlyFusionModel(nn.Module):
    def __init__(self, num_classes, metadata_dim=15, embed_dim=256, dropout=0.15):
        super(EarlyFusionModel, self).__init__()

        efficientnet = models.efficientnet_b0(weights='DEFAULT')
        self.image_encoder = nn.Sequential(*list(efficientnet.children())[:-1])

        # Metadata is projected to a smaller feature vector before concatenation.
        self.metadata_projection = nn.Sequential(
            nn.Linear(metadata_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
        )

        self.fusion_network = nn.Sequential(
            nn.Linear(1280 + 128, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(512, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(embed_dim, num_classes),
        )

    def forward(self, images, metadata):
        img_features = self.image_encoder(images)
        img_features = img_features.flatten(start_dim=1)

        meta_features = self.metadata_projection(metadata)

        fused = torch.cat([img_features, meta_features], dim=1)

        output = self.fusion_network(fused)
        return output


# ---- Quick model test ----
if __name__ == '__main__':
    num_classes = 162

    model = EarlyFusionModel(num_classes=num_classes).to(config.DEVICE)

    dummy_images   = torch.randn(4, 3, 224, 224).to(config.DEVICE)
    dummy_metadata = torch.randn(4, 15).to(config.DEVICE)

    output = model(dummy_images, dummy_metadata)

    print(f"Image input shape:    {dummy_images.shape}")
    print(f"Metadata input shape: {dummy_metadata.shape}")
    print(f"Output shape:         {output.shape}")   # expected: [4, 162]
    print(f"Device: {next(model.parameters()).device}")
    print("Early fusion model check passed!")

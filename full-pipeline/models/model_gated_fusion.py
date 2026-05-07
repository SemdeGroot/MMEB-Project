import torch
import torch.nn as nn
from torchvision import models

import config


# ---- Gated fusion ----
# A learned gate decides how much the image and metadata embeddings should contribute.

class GatedFusionModel(nn.Module):
    def __init__(self, num_classes, metadata_dim=4, embed_dim=256, dropout=0.15):
        super(GatedFusionModel, self).__init__()

        efficientnet = models.efficientnet_b0(weights='DEFAULT')

        self.image_encoder = nn.Sequential(*list(efficientnet.children())[:-1])
        self.image_projection = nn.Sequential(
            nn.Linear(1280, embed_dim),
            nn.ReLU(),
        )
        self.image_embed_dim = embed_dim

        # Both branches must end up in the same space before gating.
        self.metadata_encoder = nn.Sequential(
            nn.Linear(metadata_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(128, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(),
        )

        # The gate is predicted from both embeddings together.
        self.gate = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Sigmoid()
        )

        # Alpha blends two gated views and starts from an equal mix.
        self.alpha = nn.Parameter(torch.tensor(0.5))

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, images, metadata):
        img_embed = self.image_encoder(images)
        img_embed = img_embed.flatten(start_dim=1)
        img_embed = self.image_projection(img_embed)

        meta_embed = self.metadata_encoder(metadata)

        combined = torch.cat([img_embed, meta_embed], dim=1)
        g = self.gate(combined)

        gated_img  = g * img_embed  + (1 - g) * meta_embed
        gated_meta = (1 - g) * meta_embed + g * img_embed

        fused = self.alpha * gated_img + (1 - self.alpha) * gated_meta

        output = self.classifier(fused)

        return output, img_embed, meta_embed


# ---- Quick model test ----
if __name__ == '__main__':
    num_classes = 162

    model = GatedFusionModel(num_classes=num_classes).to(config.DEVICE)

    dummy_images   = torch.randn(4, 3, 224, 224).to(config.DEVICE)
    dummy_metadata = torch.randn(4, 4).to(config.DEVICE)

    logits, img_embed, meta_embed = model(dummy_images, dummy_metadata)

    print(f"Image input shape:    {dummy_images.shape}")
    print(f"Metadata input shape: {dummy_metadata.shape}")
    print(f"Output shape:         {logits.shape}")   # expected: [4, 162]
    print(f"Image embed shape:    {img_embed.shape}")
    print(f"Meta embed shape:     {meta_embed.shape}")
    print(f"Device: {next(model.parameters()).device}")
    print("Gated fusion model check passed!")

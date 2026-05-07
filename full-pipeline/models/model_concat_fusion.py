import torch
import torch.nn as nn
from torchvision import models

import config


# ---- Concatenation fusion ----
# Both modalities are projected into the same-sized embedding space before concatenation.

class ConcatFusionModel(nn.Module):
    def __init__(self, num_classes, metadata_dim=4, embed_dim=256, dropout=0.15):
        super(ConcatFusionModel, self).__init__()

        efficientnet = models.efficientnet_b0(weights='DEFAULT')

        # Use EfficientNet features before the original classifier head.
        self.image_encoder = nn.Sequential(*list(efficientnet.children())[:-1])
        self.image_projection = nn.Sequential(
            nn.Linear(1280, embed_dim),
            nn.ReLU(),
        )
        self.image_embed_dim = embed_dim

        # Project metadata into the same embedding size as the image branch.
        self.metadata_encoder = nn.Sequential(
            nn.Linear(metadata_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(128, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(),
        )
        self.metadata_embed_dim = embed_dim

        fusion_dim = self.image_embed_dim + self.metadata_embed_dim
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(fusion_dim, 256),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, images, metadata):
        img_embed = self.image_encoder(images)
        img_embed = img_embed.flatten(start_dim=1)
        img_embed = self.image_projection(img_embed)

        meta_embed = self.metadata_encoder(metadata)

        fused = torch.cat([img_embed, meta_embed], dim=1)

        output = self.classifier(fused)

        # Returning embeddings keeps the interface compatible with optional analysis code.
        return output, img_embed, meta_embed


# ---- Quick model test ----
if __name__ == '__main__':
    num_classes = 162

    model = ConcatFusionModel(num_classes=num_classes).to(config.DEVICE)

    dummy_images   = torch.randn(4, 3, 224, 224).to(config.DEVICE)
    dummy_metadata = torch.randn(4, 4).to(config.DEVICE)

    logits, img_embed, meta_embed = model(dummy_images, dummy_metadata)

    print(f"Image input shape:    {dummy_images.shape}")
    print(f"Metadata input shape: {dummy_metadata.shape}")
    print(f"Output shape:         {logits.shape}")   # expected: [4, 162]
    print(f"Image embed shape:    {img_embed.shape}")
    print(f"Meta embed shape:     {meta_embed.shape}")
    print(f"Device: {next(model.parameters()).device}")
    print("Concatenation fusion model check passed!")

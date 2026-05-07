import torch
import torch.nn as nn
from torchvision import models

import config


# ---- Cross-attention transformer fusion ----
# This variant explicitly lets each modality attend to the other before the
# final transformer encoder integrates the fused representation.

class CrossAttentionTransformerFusion(nn.Module):
    def __init__(self, num_classes, metadata_dim=15, embed_dim=256,
                 nhead=8, num_layers=3, dropout=0.15):
        super(CrossAttentionTransformerFusion, self).__init__()

        efficientnet = models.efficientnet_b0(weights='DEFAULT')
        self.image_encoder = nn.Sequential(*list(efficientnet.children())[:-1])
        self.image_projection = nn.Sequential(
            nn.Linear(1280, embed_dim),
            nn.LayerNorm(embed_dim),
        )

        self.metadata_encoder = nn.Sequential(
            nn.Linear(metadata_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, embed_dim),
            nn.LayerNorm(embed_dim),
        )

        self.image_to_metadata_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.metadata_to_image_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=512,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, images, metadata):
        img_embed   = self.image_encoder(images).flatten(start_dim=1)
        img_token   = self.image_projection(img_embed)
        meta_token  = self.metadata_encoder(metadata)

        img_seq  = img_token.unsqueeze(1)
        meta_seq = meta_token.unsqueeze(1)

        img_attended,  _ = self.image_to_metadata_attention(
            query=img_seq, key=meta_seq, value=meta_seq)

        meta_attended, _ = self.metadata_to_image_attention(
            query=meta_seq, key=img_seq, value=img_seq)

        combined = img_attended + meta_attended

        out     = self.transformer(combined)
        cls_out = out.squeeze(1)

        logits = self.classifier(cls_out)

        return logits, img_token, meta_token


# ---- Quick model test ----
if __name__ == '__main__':
    num_classes = 162

    model = CrossAttentionTransformerFusion(num_classes=num_classes).to(config.DEVICE)

    dummy_images   = torch.randn(4, 3, 224, 224).to(config.DEVICE)
    dummy_metadata = torch.randn(4, 15).to(config.DEVICE)

    logits, z_img, z_meta = model(dummy_images, dummy_metadata)

    print(f"Image input shape:    {dummy_images.shape}")
    print(f"Metadata input shape: {dummy_metadata.shape}")
    print(f"Output shape:         {logits.shape}")    # expected: [4, 162]
    print(f"Image embed shape:    {z_img.shape}")     # expected: [4, 256]
    print(f"Meta embed shape:     {z_meta.shape}")    # expected: [4, 256]
    print(f"Device: {next(model.parameters()).device}")
    print("Cross-attention transformer model check passed!")

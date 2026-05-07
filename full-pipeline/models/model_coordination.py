import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

import config


# ---- Coordination contrastive loss ----
# Here, the positive pair is the image and metadata from the same occurrence.
# Other image-metadata combinations in the batch act as negatives.

class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature

    def forward(self, z_img, z_meta):
        batch_size = z_img.shape[0]

        z_img  = F.normalize(z_img,  dim=1)
        z_meta = F.normalize(z_meta, dim=1)

        z = torch.cat([z_img, z_meta], dim=0)

        sim_matrix = torch.matmul(z, z.T) / self.temperature

        mask = torch.eye(2 * batch_size, device=z.device).bool()
        sim_matrix = sim_matrix.masked_fill(mask, float('-inf'))

        labels = torch.arange(batch_size, device=z.device)
        labels = torch.cat([labels + batch_size, labels], dim=0)

        loss = F.cross_entropy(sim_matrix, labels)
        return loss


# ---- Transformer fusion with coordination ----
# This model shares the same image/metadata encoders as the transformer fusion model,
# but it also returns pre-fusion embeddings for the contrastive loss.

class TransformerFusionWithCoordination(nn.Module):
    def __init__(self, num_classes, metadata_dim=4, embed_dim=256, nhead=4, num_layers=2, dropout=0.15):
        super(TransformerFusionWithCoordination, self).__init__()

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
            nn.Dropout(p=dropout),
            nn.Linear(128, embed_dim),
            nn.LayerNorm(embed_dim),
        )

        # The transformer integrates a class token with image and metadata tokens.
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.modality_embeddings = nn.Parameter(torch.zeros(1, 3, embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=512,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, images, metadata):
        img_embed  = self.image_encoder(images)
        img_embed  = img_embed.flatten(start_dim=1)
        img_token  = self.image_projection(img_embed)
        meta_token = self.metadata_encoder(metadata)

        batch_size = images.size(0)
        cls_token = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.stack([img_token, meta_token], dim=1)
        tokens = torch.cat([cls_token, tokens], dim=1)
        tokens = tokens + self.modality_embeddings
        out    = self.transformer(tokens)
        cls_out = out[:, 0]

        logits = self.classifier(cls_out)

        # The returned embeddings are the pre-fusion representations used by NT-Xent.
        return logits, img_token, meta_token


# ---- Quick model test ----
if __name__ == '__main__':
    num_classes = 162

    model       = TransformerFusionWithCoordination(num_classes=num_classes).to(config.DEVICE)
    contrastive = NTXentLoss(temperature=0.5).to(config.DEVICE)

    dummy_images   = torch.randn(4, 3, 224, 224).to(config.DEVICE)
    dummy_metadata = torch.randn(4, 4).to(config.DEVICE)
    dummy_labels   = torch.randint(0, num_classes, (4,)).to(config.DEVICE)

    # Forward pass returns logits + embeddings
    logits, z_img, z_meta = model(dummy_images, dummy_metadata)

    # Classification loss
    cls_loss  = nn.CrossEntropyLoss()(logits, dummy_labels)

    # Contrastive loss
    con_loss  = contrastive(z_img, z_meta)

    # Total loss
    lam       = 0.5   # lambda: weight of contrastive loss
    total     = cls_loss + lam * con_loss

    print(f"Output shape:        {logits.shape}")    # [4, 162]
    print(f"Image embed shape:   {z_img.shape}")     # [4, 256]
    print(f"Meta embed shape:    {z_meta.shape}")    # [4, 256]
    print(f"Classification loss: {cls_loss.item():.4f}")
    print(f"Contrastive loss:    {con_loss.item():.4f}")
    print(f"Total loss:          {total.item():.4f}")
    print(f"Device: {next(model.parameters()).device}")
    print("Coordination model check passed!")

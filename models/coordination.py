import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm
import open_clip


class NTXentLoss(nn.Module):
    """Normalized Temperature-scaled Cross Entropy loss for contrastive alignment."""

    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_img, z_meta):
        batch_size = z_img.shape[0]
        z_img  = F.normalize(z_img,  dim=1)
        z_meta = F.normalize(z_meta, dim=1)
        z      = torch.cat([z_img, z_meta], dim=0)
        sim    = torch.matmul(z, z.T) / self.temperature
        mask   = torch.eye(2 * batch_size, device=z.device).bool()
        sim    = sim.masked_fill(mask, float('-inf'))
        labels = torch.cat([
            torch.arange(batch_size, device=z.device) + batch_size,
            torch.arange(batch_size, device=z.device),
        ])
        return F.cross_entropy(sim, labels)


class CoordinationModel(nn.Module):
    """Transformer fusion with contrastive alignment between image and metadata embeddings.

    The classification loss is combined with NT-Xent during training (LAMBDA=0.05).
    The model returns (logits, img_token, meta_token) so the training loop can
    compute the contrastive term.
    """

    def __init__(self, num_classes, metadata_dim, backbone="resnet50",
                 embed_dim=256, nhead=4, num_layers=2, dropout=0.15):
        super().__init__()
        if backbone == "resnet50":
            net = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V1)
            net.fc = nn.Identity()
            self.backbone = net
            img_dim = 2048
        elif backbone == "bioclip":
            clip_model, _, _ = open_clip.create_model_and_transforms("hf-hub:imageomics/bioclip")
            self.backbone = clip_model.visual
            img_dim = 512
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

        self.image_projection = nn.Sequential(
            nn.Linear(img_dim, embed_dim), nn.LayerNorm(embed_dim),
        )
        self.metadata_encoder = nn.Sequential(
            nn.Linear(metadata_dim, 128),
            nn.LayerNorm(128), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(128, embed_dim), nn.LayerNorm(embed_dim),
        )

        self.cls_token          = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.modality_embeddings = nn.Parameter(torch.zeros(1, 3, embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=nhead, dim_feedforward=512,
            dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, img, meta):
        img_token  = self.image_projection(self.backbone(img))
        meta_token = self.metadata_encoder(meta)

        batch_size = img.size(0)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        tokens     = torch.stack([img_token, meta_token], dim=1)
        tokens     = torch.cat([cls_tokens, tokens], dim=1)
        tokens     = tokens + self.modality_embeddings
        out        = self.transformer(tokens)
        logits     = self.classifier(out[:, 0])

        return logits, img_token, meta_token

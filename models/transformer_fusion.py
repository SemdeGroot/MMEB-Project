import torch
import torch.nn as nn
import torchvision.models as tvm
import open_clip


class TransformerFusionModel(nn.Module):
    """Bidirectional cross-attention between image and metadata tokens,
    followed by a small transformer encoder."""

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

        # Each modality attends to the other before the transformer encoder.
        self.image_to_metadata_attn = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=nhead, dropout=dropout, batch_first=True)
        self.metadata_to_image_attn = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=nhead, dropout=dropout, batch_first=True)

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

        img_seq  = img_token.unsqueeze(1)
        meta_seq = meta_token.unsqueeze(1)

        img_attended,  _ = self.image_to_metadata_attn(
            query=img_seq, key=meta_seq, value=meta_seq)
        meta_attended, _ = self.metadata_to_image_attn(
            query=meta_seq, key=img_seq, value=img_seq)

        combined = img_attended + meta_attended
        out      = self.transformer(combined)
        logits   = self.classifier(out.squeeze(1))

        return logits, img_token, meta_token

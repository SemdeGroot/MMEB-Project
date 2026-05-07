import torch
import torch.nn as nn
import torchvision.models as tvm
import open_clip


class ConcatFusionModel(nn.Module):
    """Both modalities projected to same embedding size, then concatenated."""

    def __init__(self, num_classes, metadata_dim, backbone="resnet50",
                 embed_dim=256, dropout=0.15):
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
            nn.Linear(img_dim, embed_dim), nn.ReLU(),
        )
        self.metadata_encoder = nn.Sequential(
            nn.Linear(metadata_dim, 128),
            nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, embed_dim),
            nn.BatchNorm1d(embed_dim), nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, img, meta):
        img_embed  = self.image_projection(self.backbone(img))
        meta_embed = self.metadata_encoder(meta)
        fused      = torch.cat([img_embed, meta_embed], dim=1)
        return self.classifier(fused), img_embed, meta_embed

import torch
import torch.nn as nn
import torchvision.models as tvm
import open_clip


class EarlyFusionModel(nn.Module):
    """Concatenates image and location features before a single classifier head."""

    def __init__(self, num_classes, backbone="resnet50", metadata_dim=4):
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

        self.loc_encoder = nn.Sequential(
            nn.Linear(metadata_dim, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
        )
        self.head = nn.Linear(img_dim + 64, num_classes)

    def forward(self, img, loc):
        img_feat = self.backbone(img)
        loc_feat = self.loc_encoder(loc)
        return self.head(torch.cat([img_feat, loc_feat], dim=1))

import torch
import torch.nn as nn
import torchvision.models as tvm
import open_clip


class GatedFusionModel(nn.Module):
    """Gated Multimodal Unit fusion (Arevalo et al., ICLR Workshop 2017).

    Per fused-feature dimension a gate decides how much of the image vs location
    contributes. The gate is computed from both modalities so neither one can
    silently dominate without it showing in the gate statistics.
    """

    def __init__(self, num_classes, backbone="resnet50", fused_dim=512):
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
            nn.Linear(4, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
        )

        self.img_proj = nn.Linear(img_dim, fused_dim)
        self.loc_proj = nn.Linear(64,      fused_dim)
        self.gate     = nn.Linear(img_dim + 64, fused_dim)

        self.head = nn.Linear(fused_dim, num_classes)

    def forward(self, img, loc):
        img_feat = self.backbone(img)
        loc_feat = self.loc_encoder(loc)

        h_img = torch.tanh(self.img_proj(img_feat))
        h_loc = torch.tanh(self.loc_proj(loc_feat))
        z     = torch.sigmoid(self.gate(torch.cat([img_feat, loc_feat], dim=1)))

        fused = z * h_img + (1 - z) * h_loc
        return self.head(fused)

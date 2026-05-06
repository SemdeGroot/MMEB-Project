import torch.nn as nn
import torchvision.models as tvm
import open_clip


class BaselineModel(nn.Module):
    """Image-only classifier: ResNet-50 or BioCLIP backbone + Linear head."""

    def __init__(self, num_classes, backbone="resnet50"):
        super().__init__()
        if backbone == "resnet50":
            net = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V1)
            net.fc = nn.Identity()
            self.backbone = net
            feat_dim = 2048
        elif backbone == "bioclip":
            clip_model, _ = open_clip.create_model_and_transforms("hf-hub:imageomics/bioclip")
            self.backbone = clip_model.visual
            feat_dim = 512
        else:
            raise ValueError(f"Unknown backbone: {backbone}")
        self.head = nn.Linear(feat_dim, num_classes)

    def forward(self, img, loc):
        # loc is ignored — image-only model
        return self.head(self.backbone(img))

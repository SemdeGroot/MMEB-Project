import torch.nn as nn


class LocationOnlyModel(nn.Module):
    """GPS + timestamp only. no image input. Upper bound on location signal."""

    def __init__(self, num_classes):
        super().__init__()
        self.loc_encoder = nn.Sequential(
            nn.Linear(4, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
        )
        self.head = nn.Linear(64, num_classes)

    def forward(self, img, loc):
        # img is ignored — location-only model
        return self.head(self.loc_encoder(loc))

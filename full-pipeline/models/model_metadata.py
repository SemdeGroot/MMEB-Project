import torch
import torch.nn as nn

import config


# ---- Metadata-only baseline ----
# A small MLP is enough here because the input is already a processed feature vector.

class MetadataOnlyModel(nn.Module):
    def __init__(self, num_classes, input_dim=15, hidden_dim=256, dropout=0.2):
        super(MetadataOnlyModel, self).__init__()

        # The encoder first expands the feature space, then compresses it again.
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout + 0.05),

            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout + 0.05),

            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, images=None, metadata=None):
        # images is ignored so the model can share the same call signature as fusion models.
        x = self.encoder(metadata)
        return self.classifier(x)


# ---- Quick model test ----
if __name__ == '__main__':
    num_classes = 162

    model = MetadataOnlyModel(num_classes=num_classes, input_dim=15).to(config.DEVICE)

    # Dummy inputs (input_dim=15 matches the expanded metadata feature set)
    dummy_images   = torch.randn(4, 3, 224, 224).to(config.DEVICE)  # ignored
    dummy_metadata = torch.randn(4, 15).to(config.DEVICE)

    output = model(dummy_images, dummy_metadata)

    print(f"Input shape:  {dummy_metadata.shape}")
    print(f"Output shape: {output.shape}")   # expected: [4, 162]
    print(f"Device: {next(model.parameters()).device}")
    print("Metadata model check passed!")

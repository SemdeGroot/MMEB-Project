import torch.nn as nn


class MetadataOnlyModel(nn.Module):
    """Deep MLP on rich metadata features only. Image input is ignored."""

    def __init__(self, num_classes, metadata_dim, hidden_dim=256, dropout=0.2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(metadata_dim, hidden_dim),
            nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2), nn.GELU(), nn.Dropout(dropout + 0.05),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2), nn.GELU(), nn.Dropout(dropout + 0.05),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(dropout),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, img, meta):
        return self.classifier(self.encoder(meta))

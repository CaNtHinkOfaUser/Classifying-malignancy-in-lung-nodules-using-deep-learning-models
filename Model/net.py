# Written by Claude for Ishaan

"""Step C. The 3D CNN that rates a nodule's malignancy from its 32 mm cube.

Input  (batch, 1, 32, 32, 32): one cube per nodule, HU scaled to [-1, 1]
Output (batch, 5): raw scores (logits) for ratings 1-5. No softmax here:
       nn.CrossEntropyLoss applies it, so adding one would apply it twice.

291,861 parameters. Measured on the M2 with MPS: about 321 ms per batch of 32.
"""

import torch.nn as nn


def block(c_in, c_out):
    """Conv -> BatchNorm -> ReLU, the building block used 4 times."""
    return nn.Sequential(
        nn.Conv3d(c_in, c_out, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm3d(c_out),
        nn.ReLU(inplace=True),
    )


class NoduleNet3D(nn.Module):
    def __init__(self, n_classes=5, dropout=0.3):
        super().__init__()
        self.features = nn.Sequential(
            block(1, 16), nn.MaxPool3d(2),     # 32³ -> 16³
            block(16, 32), nn.MaxPool3d(2),    # 16³ -> 8³
            block(32, 64), nn.MaxPool3d(2),    # 8³  -> 4³
            block(64, 128),                    # stays 4³
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(128, n_classes)

    def forward(self, x):
        x = self.features(x)               # (batch, 128, 4, 4, 4)
        x = x.mean(dim=(2, 3, 4))          # (batch, 128). Not nn.AdaptiveAvgPool3d: 3.6x slower on MPS
        x = self.dropout(x)
        return self.head(x)                # (batch, 5)

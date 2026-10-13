# Ishaan

"""Step C. The 3D CNN that rates a nodule's malignancy from its cube.

Input  (batch, in_channels, n, n, n): one cube per nodule, n = the Dataset's CROP
       (40 in no_folds, 32 in folds; any multiple of 8 works).
       channel 0: HU scaled to [-1, 1]
       channel 1 (in_channels=2 only): the nodule's box mask, 0/1 (Dataset mask=True)
Output (batch, 5): raw scores (logits) for ratings 1-5. No softmax here:
       nn.CrossEntropyLoss applies it, so adding one would apply it twice.

291,861 parameters (292,293 with in_channels=2: the first conv's 16 filters each
get 27 weights for the 2nd channel). Measured on the M2 with MPS: about 321 ms per
batch of 32 at n = 32.
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
    def __init__(self, n_classes=5, dropout=0.3, in_channels=1):
        # in_channels: 1 = CT only (every run so far), 2 = CT + box mask.
        # It must match the checkpoint: a 1-channel best.pt won't load into a 2-channel net.
        super().__init__()
        self.features = nn.Sequential(
            block(in_channels, 16), nn.MaxPool3d(2),   # n³   -> (n/2)³, e.g. 40³ -> 20³
            block(16, 32), nn.MaxPool3d(2),            # (n/2)³ -> (n/4)³
            block(32, 64), nn.MaxPool3d(2),            # (n/4)³ -> (n/8)³
            block(64, 128),                            # stays (n/8)³
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(128, n_classes)

    def forward(self, x):
        x = self.features(x)               # (batch, 128, n/8, n/8, n/8): 5³ for a 40 crop
        x = x.mean(dim=(2, 3, 4))          # (batch, 128). Not nn.AdaptiveAvgPool3d: 3.6x slower on MPS
        x = self.dropout(x)
        return self.head(x)                # (batch, 5)

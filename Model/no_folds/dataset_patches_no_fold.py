"""PyTorch Dataset for the 3D CNN, using ONE train / val / test split (no folds).

The same as Model/folds/dataset_patches_fold.py except for where the split comes
from: this file reads patient_split.csv (70 / 15 / 15), the folds version reads the folds.

Reads:      data/nodules_labels.csv      labels (hard, soft_1..5) and patch_file
            data/patient_split.csv       which split each patient is in (source of truth)
            PATCH_DIR/<nodule_key>.npy   int16 HU, (48, 48, 48), axes (z, row, col), 1 mm voxels
            MASK_DIR/<nodule_key>.npy    mask=True only: uint8 0/1, the same grid (4bbuild_patches.py)
Returns:    (x, target, i) per nodule, the same shape as the folds version, so train.py's loop fits
                x       float32 (1, CROP, CROP, CROP), HU mapped to -1..1 by a fixed window
                        mask=True: (2, CROP, CROP, CROP), channel 1 the box mask as 0/1
                target  labels="hard": int64 0..4, the median rating minus 1   (model A)
                        labels="soft": float32 (5,), share of readers per rating (model B)
                i       the row in self.df, which holds nodule_key and everything else
Usage:      from dataset_patches_no_fold import NoduleDataset
            train = NoduleDataset("train", "hard")    # augmented automatically
            val = NoduleDataset("val", "hard")        # val and test are never augmented
            train = NoduleDataset("train", "hard", mask=True)   # for NoduleNet3D(in_channels=2)
Check:      .venv/bin/python Model/no_folds/dataset_patches_no_fold.py   (needs the external drive mounted)

A and B read the same Dataset and differ only in `labels`, so the input
pipeline can't be a difference between the two models.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# let this file find pipeline/ct.py, whichever folder it's run from.
# This file is in Model/<folder>/, so the project folder is three levels up: parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "pipeline"))
from ct import CUBE_MM, CUBE_SPACING, MASK_DIR, NODULES, PATCH_DIR, SPLIT

SPLITS = ("train", "val", "test")
LABELS = ("hard", "soft")
SOFT_COLS = [f"soft_{k}" for k in range(1, 6)]

CUBE = int(round(CUBE_MM / CUBE_SPACING))   # side of the saved cube in voxels: 48
CROP = 40         # side of the cube the model sees. net.py works with any multiple of 8 up to 48
MAX_SHIFT = 4     # train only: the crop's centre can move up to 4 mm each way

# HU window for the classifier, not the YOLO lung window. Air (-1000) is -1,
# soft tissue (~40) is ~0.49, and anything from 400 HU up is 1, so a calcified
# nodule (a sign of benign disease) still stands out from a solid one.
CNN_WINDOW = (-1000, 400)


def scale_hu(cube, window=CNN_WINDOW):
    """Fixed HU -> -1..1. The same HU is the same value in every cube (no per-cube min/max)."""
    lo, hi = window
    return np.clip((cube.astype(np.float32) - lo) / (hi - lo), 0.0, 1.0) * 2 - 1


def crop(cube, start):
    """The CROP x CROP x CROP part of the cube that starts at `start` = (z, row, col).

    Works on the last three axes, so a (channel, z, row, col) stack is cut in one go.
    """
    z, r, c = start
    return cube[..., z:z + CROP, r:r + CROP, c:c + CROP]


def random_start():
    """A crop start up to MAX_SHIFT voxels (= mm) from the centre, never past the cube's edge."""
    centre = (CUBE - CROP) // 2
    lo, hi = max(centre - MAX_SHIFT, 0), min(centre + MAX_SHIFT, CUBE - CROP)
    return [int(s) for s in torch.randint(lo, hi + 1, (3,))]


def augment(cube):
    """Random flip on each axis, then a random 0/90/180/270 degree turn in the axial plane.

    Only moves voxels around: no value changes and no resizing, so the nodule's
    size and density are untouched. Valid because the voxels are 1 mm in every
    direction. Uses torch's random numbers, not numpy's: DataLoader workers each
    get their own torch seed, but would all copy the same numpy one.

    Works on the last three axes (z, row, col), so on a (channel, z, row, col)
    stack every channel gets the same flips and turn: the mask stays on the nodule.
    """
    for axis in (-3, -2, -1):
        if torch.rand(1).item() < 0.5:
            cube = np.flip(cube, axis=axis)
    cube = np.rot90(cube, k=int(torch.randint(4, (1,)).item()), axes=(-2, -1))
    return np.ascontiguousarray(cube)     # torch can't use numpy's flipped views


class NoduleDataset(Dataset):
    def __init__(self, split, labels, augment=None, preload=True, mask=False):
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}, not {split!r}")
        if labels not in LABELS:
            raise ValueError(f"labels must be one of {LABELS}, not {labels!r}")

        nodules = pd.read_csv(NODULES)
        split_of = pd.read_csv(SPLIT).set_index("patient_id")["split"]
        frozen = nodules["patient_id"].map(split_of)
        if frozen.isna().any():
            raise ValueError("some patients in nodules_labels.csv are missing from patient_split.csv")
        if "split" in nodules.columns and not (nodules["split"] == frozen).all():
            raise ValueError("nodules_labels.csv disagrees with patient_split.csv: rerun the split step")

        self.df = nodules[frozen == split].reset_index(drop=True)
        self.paths = [PATCH_DIR / f for f in self.df["patch_file"]]
        missing = [p for p in self.paths if not p.exists()]
        if missing:
            raise FileNotFoundError(f"{len(missing)} of {len(self.paths)} {split} patches not found, "
                                    f"e.g. {missing[0]}. Is the external drive mounted?")

        # the box masks have the same file names as the patches, in their own folder
        self.mask = mask
        self.mask_paths = [MASK_DIR / f for f in self.df["patch_file"]] if mask else None
        if mask:
            missing = [p for p in self.mask_paths if not p.exists()]
            if missing:
                raise FileNotFoundError(f"{len(missing)} of {len(self.mask_paths)} {split} masks not found, "
                                        f"e.g. {missing[0]}. Run: python pipeline/4bbuild_patches.py --masks-only")

        self.labels = labels
        self.hard = torch.tensor(self.df["hard"].to_numpy() - 1, dtype=torch.long)
        self.soft = torch.tensor(self.df[SOFT_COLS].to_numpy(), dtype=torch.float32)
        self.augment = (split == "train") if augment is None else augment    # only train by default

        # One cube is 48^3 int16 = 0.2 MB, so all of train is ~0.3 GB. Holding them
        # in memory saves re-reading every file over USB on every epoch.
        self.cubes = [np.load(p) for p in self.paths] if preload else None
        # a mask is 48^3 uint8 = 0.1 MB, so ~0.15 GB more for train
        self.masks = [np.load(p) for p in self.mask_paths] if (mask and preload) else None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        cube = self.cubes[i] if self.cubes is not None else np.load(self.paths[i])
        if self.mask:
            # (2, 48, 48, 48): CT and mask as ONE array, so the crop and augment
            # below move both by the same random amounts
            m = self.masks[i] if self.masks is not None else np.load(self.mask_paths[i])
            cube = np.stack([cube, m.astype(cube.dtype)])

        # 1. crop: exactly the centre for val and test, shifted a little for train
        if self.augment:
            cube = crop(cube, random_start())
            cube = augment(cube)          # 2. flips and turns, train only
        else:
            cube = crop(cube, [(CUBE - CROP) // 2] * 3)

        # 3. HU -> -1..1 for the CT only (the mask stays 0/1), as (channels, CROP, CROP, CROP)
        if self.mask:
            x = torch.from_numpy(np.stack([scale_hu(cube[0]), cube[1].astype(np.float32)]))
        else:
            x = torch.from_numpy(scale_hu(cube)).unsqueeze(0)

        # 4. the target: the ONLY difference between model A and model B
        target = self.hard[i] if self.labels == "hard" else self.soft[i]
        return x, target, i


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))    # Model/, where the shared net.py is
    from net import NoduleNet3D

    sets = {s: NoduleDataset(s, "hard") for s in SPLITS}
    total = len(pd.read_csv(NODULES))
    print("nodules per split:", {s: len(d) for s, d in sets.items()}, f"(MUST sum to {total})")
    patients = [set(d.df["patient_id"]) for d in sets.values()]
    overlap = (patients[0] & patients[1]) | (patients[0] & patients[2]) | (patients[1] & patients[2])
    print(f"patients in more than one split: {len(overlap)} (MUST be 0)")
    print(f"augmented: { {s: d.augment for s, d in sets.items()} } (MUST be train only)")

    x, target, i = sets["val"][0]
    print(f"one item: {sets['val'].df['nodule_key'][i]}  x {tuple(x.shape)} {x.dtype} "
          f"range {x.min():.3f}..{x.max():.3f}  hard target {int(target)}")
    print(f"soft target sums to 1: {float(NoduleDataset('val', 'soft')[0][1].sum()):.3f}")
    print(f"hard labels within 0..4: {all(0 <= int(h) <= 4 for d in sets.values() for h in d.hard)}")
    print(f"val is unchanged between reads: {torch.equal(sets['val'][0][0], sets['val'][0][0])}")

    # augmentation must only move voxels: same values, rearranged
    raw = crop(sets["train"].cubes[0], [(CUBE - CROP) // 2] * 3)
    print(f"flips/turns keep every voxel value: "
          f"{np.array_equal(np.sort(raw, axis=None), np.sort(augment(raw), axis=None))}")

    x, target, idx = next(iter(DataLoader(sets["train"], batch_size=8, shuffle=True)))
    out = NoduleNet3D()(x)
    print(f"one batch: x {tuple(x.shape)}  target {tuple(target.shape)}  -> net.py output {tuple(out.shape)}")

    # with the mask: after the random crop and augment, the mask must still sit on the
    # nodule, so the CT inside it is much denser than the (mostly lung) CT outside it
    if MASK_DIR.exists():
        masked = NoduleDataset("train", "hard", preload=False, mask=True)
        x = torch.stack([masked[i][0] for i in range(64)])
        inside, outside = x[:, 0][x[:, 1] == 1].mean(), x[:, 0][x[:, 1] == 0].mean()
        print(f"mask: x {tuple(x.shape[1:])}, channel 1 values {x[:, 1].unique().tolist()} (MUST be [0.0, 1.0])")
        print(f"mask: CT inside {inside:.2f} vs outside {outside:.2f} (inside MUST be much higher)")
        print(f"mask: -> net.py output {tuple(NoduleNet3D(in_channels=2)(x[:8]).shape)}")

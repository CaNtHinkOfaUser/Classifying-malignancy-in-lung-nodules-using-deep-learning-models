# Ishaan
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# let this file find pipeline/ct.py, whichever folder it's run from.
# This file is in Model/<folder>/, so the project folder is three levels up: parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "pipeline"))
from ct import MASK_DIR, NODULES, PATCH_DIR, fold_role

SOFT_COLS = ["soft_1", "soft_2", "soft_3", "soft_4", "soft_5"]
CROP = 40          # side of the cube the model sees (the file is 48)
MAX_SHIFT = 4      # train only: the centre can move up to 4 mm each way


class NoduleDataset(Dataset):
    def __init__(self, val_fold, part, labels, seed=0, mask=False):
        # val_fold: which cross-validation run (1-4), named after the fold it validates on.
        # part: "train" (the other three of folds 1-4), "val" (fold val_fold) or "test" (fold 0, locked).
        # mask: also return the box mask as a 2nd channel (for NoduleNet3D(in_channels=2))
        assert part in ("train", "val", "test") and labels in ("hard", "soft")
        df = pd.read_csv(NODULES)
        role = df["fold"].map(lambda f: fold_role(f, val_fold))
        self.df = df[role == part].reset_index(drop=True)
        # every cube of this split, loaded once: shape (N, 48, 48, 48), HU
        self.cubes = np.stack([np.load(PATCH_DIR / f) for f in self.df["patch_file"]])
        # the box masks, same file names in their own folder: (N, 48, 48, 48), 0/1
        self.masks = np.stack([np.load(MASK_DIR / f) for f in self.df["patch_file"]]) if mask else None
        # stored as the model's targets, as in dataset_patches_no_fold.py
        self.hard = torch.tensor(self.df["hard"].to_numpy() - 1, dtype=torch.long)    # class 0..4 = rating 1..5
        self.soft = torch.tensor(self.df[SOFT_COLS].to_numpy(), dtype=torch.float32)  # share of readers per rating
        self.labels = labels
        self.augment = (part == "train")          # only train gets random changes
        # Randomness: torch's global generator, seeded by train.py's torch.manual_seed(seed),
        # drawn in the same order as Model/no_folds/dataset_patches_no_fold.py, so a seed
        # means the same thing in both experiments. `seed` is kept so old calls still work.

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        # 0. with the mask: CT and mask as ONE (2, 48, 48, 48) array, so steps 1 and 2
        #    move both by the same random amounts. Steps 1 and 2 use the last three axes.
        x = self.cubes[i]
        if self.masks is not None:
            x = np.stack([x, self.masks[i].astype(x.dtype)])

        # 1. crop 40 out of 48; a start of 4 is the exact centre
        start_centre = (48 - CROP) // 2
        if self.augment:
            lo, hi = max(start_centre - MAX_SHIFT, 0), min(start_centre + MAX_SHIFT, 48 - CROP)
            o = [int(s) for s in torch.randint(lo, hi + 1, (3,))]
        else:
            o = [start_centre, start_centre, start_centre]
        x = x[..., o[0]:o[0] + CROP, o[1]:o[1] + CROP, o[2]:o[2] + CROP]

        # 2. augment: random flips on each axis, random quarter-turn
        if self.augment:
            for axis in (-3, -2, -1):
                if torch.rand(1).item() < 0.5:
                    x = np.flip(x, axis=axis)
            x = np.rot90(x, k=int(torch.randint(4, (1,)).item()), axes=(-2, -1))

        # 3. normalise the CT: HU [-1000, 400] -> [-1, 1] (the mask stays 0/1),
        #    then a (1, 40, 40, 40) tensor, or (2, 40, 40, 40) with the mask
        if self.masks is not None:
            x = np.stack([(np.clip(x[0], -1000, 400) + 300) / 700, x[1]])
        else:
            x = ((np.clip(x, -1000, 400) + 300) / 700)[None]
        x = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))

        # 4. the target: the ONLY difference between the two models
        target = self.hard[i] if self.labels == "hard" else self.soft[i]
        return x, target, i

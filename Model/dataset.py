# Ishaan
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# let this file find pipeline/ct.py, whichever folder it's run from
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))
from ct import NODULES, PATCH_DIR, fold_role

SOFT_COLS = ["soft_1", "soft_2", "soft_3", "soft_4", "soft_5"]
CROP = 32          # side of the cube the model sees (the file is 48)
MAX_SHIFT = 4      # train only: the centre can move up to 4 mm each way


class NoduleDataset(Dataset):
    def __init__(self, val_fold, part, labels, seed=0):
        # val_fold: which cross-validation run (1-4), named after the fold it validates on.
        # part: "train" (the other three of folds 1-4), "val" (fold val_fold) or "test" (fold 0, locked).
        assert part in ("train", "val", "test") and labels in ("hard", "soft")
        df = pd.read_csv(NODULES)
        role = df["fold"].map(lambda f: fold_role(f, val_fold))
        self.df = df[role == part].reset_index(drop=True)
        # every cube of this split, loaded once: shape (N, 48, 48, 48), HU
        self.cubes = np.stack([np.load(PATCH_DIR / f) for f in self.df["patch_file"]])
        self.hard = self.df["hard"].to_numpy()
        self.soft = self.df[SOFT_COLS].to_numpy()
        self.labels = labels
        self.augment = (part == "train")          # only train gets random changes
        self.rng = np.random.default_rng(seed)     # randomness follows the run's seed

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        # 1. crop 32 out of 48; a start of 8 is the exact centre
        if self.augment:
            o = self.rng.integers(8 - MAX_SHIFT, 8 + MAX_SHIFT + 1, size=3)
        else:
            o = [8, 8, 8]
        x = self.cubes[i][o[0]:o[0] + CROP, o[1]:o[1] + CROP, o[2]:o[2] + CROP]

        # 2. augment: random flips on each axis, random quarter-turn
        if self.augment:
            for axis in range(3):
                if self.rng.random() < 0.5:
                    x = np.flip(x, axis=axis)
            x = np.rot90(x, k=self.rng.integers(0, 4), axes=(1, 2))

        # 3. normalise: HU [-1000, 400] -> [-1, 1], then a (1, 32, 32, 32) tensor
        x = (np.clip(x, -1000, 400) + 300) / 700
        x = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))[None]

        # 4. the target: the ONLY difference between the two models
        if self.labels == "hard":
            target = torch.tensor(self.hard[i] - 1, dtype=torch.long)
        else:
            target = torch.tensor(self.soft[i], dtype=torch.float32)
        return x, target, i

"""Predict the locked TEST fold (fold 0) with every finished cross-validation run. Run this once, at the very end.

For each run folder (hard_fold1 ... soft_fold4) it loads best.pt, the weights of
the epoch with the lowest loss on that run's validation fold, predicts the 377
fold-0 nodules from the exact-centre crop (no augmentation), and writes
test_preds.csv next to val_preds.csv:
    nodule_key, p1..p5        softmax probabilities of ratings 1-5

It only predicts; it scores nothing. Scoring is score_test_fold.py, a separate,
deliberate step. It will not overwrite an existing test_preds.csv unless you pass
--force, so the test fold can't be predicted twice by accident.

Run from the project folder, with the patches available (on Google Colab, set
LIDC_PATCH_DIR first, as for training):
    .venv/bin/python Model/folds/predict_test_fold.py --runs-dir runs/cls
    .venv/bin/python Model/folds/predict_test_fold.py --runs-dir runs/cls_mask
Each run's config.json says whether it was trained with the box mask ("mask": true),
and the Dataset and network are built to match.
On Colab:
    !python Model/folds/predict_test_fold.py --runs-dir "{RUNS_DIR}"
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))        # Model/folds/
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))    # Model/: the shared net.py
import dataset_patches_fold
from dataset_patches_fold import CROP, NoduleDataset
from net import NoduleNet3D
from train import run_epoch

PROJECT = Path(__file__).resolve().parents[2]
RUNS = [f"{labels}_fold{k}" for k in (1, 2, 3, 4) for labels in ("hard", "soft")]


def parse_args():
    p = argparse.ArgumentParser(description="Predict the locked test fold with each finished cross-validation run.")
    p.add_argument("--runs-dir", type=Path, default=PROJECT / "runs" / "cls",
                   help="the folder holding hard_fold1 ... soft_fold4")
    p.add_argument("--runs", nargs="+", default=RUNS, help="which run folders to predict (default: all eight)")
    p.add_argument("--force", action="store_true", help="overwrite an existing test_preds.csv")
    p.add_argument("--labels-file", type=Path, default=None,
                   help="nodules_labels.csv whose `fold` column the runs were trained with "
                        "(default: data/nodules_labels.csv; the first folds: data/old_folds_v1/nodules_labels.csv)")
    return p.parse_args()


def main():
    args = parse_args()
    if args.labels_file:                   # the Dataset reads its folds from this file
        dataset_patches_fold.NODULES = args.labels_file
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # ---- check every run before predicting any ----
    run_dirs = [args.runs_dir / name for name in args.runs]
    for run_dir in run_dirs:
        if "overfit" in run_dir.name:
            sys.exit(f"{run_dir.name} is a memorise check, not a real run: leave it out")
        for f in ("config.json", "best.pt", "val_preds.csv"):
            if not (run_dir / f).exists():
                sys.exit(f"{run_dir / f} not found: has this run finished?")
        cfg = json.loads((run_dir / "config.json").read_text())
        if cfg.get("overfit"):
            sys.exit(f"{run_dir.name} is a memorise check, not a real run: leave it out")
        if "crop" in cfg and cfg["crop"] != CROP:
            sys.exit(f"{run_dir.name} was trained on {cfg['crop']} mm crops, but the Dataset now cuts "
                     f"{CROP} mm (CROP in dataset_patches_fold.py). Set it back before predicting.")
        if (run_dir / "test_preds.csv").exists() and not args.force:
            sys.exit(f"{run_dir / 'test_preds.csv'} already exists. The test fold is predicted once; "
                     "pass --force only if you really mean to replace it.")

    # ---- predict ----
    test_sets = {}       # one Dataset per (label type, mask): every run's test fold is the same fold 0
    loss_fn = nn.CrossEntropyLoss()
    for run_dir in run_dirs:
        cfg = json.loads((run_dir / "config.json").read_text())
        labels = cfg["labels"]
        mask = cfg.get("mask", False)     # runs from before --mask existed have no entry: CT only
        if (labels, mask) not in test_sets:
            test_sets[labels, mask] = NoduleDataset(cfg["fold"], "test", labels, mask=mask)   # never augmented
        test_ds = test_sets[labels, mask]
        loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=0)

        model = NoduleNet3D(in_channels=2 if mask else 1).to(device)
        model.load_state_dict(torch.load(run_dir / "best.pt", map_location=device))
        _, probs, idxs = run_epoch(model, loader, loss_fn, device)      # no optimizer: scores only

        preds = pd.DataFrame(probs, columns=[f"p{k}" for k in range(1, 6)])
        preds.insert(0, "nodule_key", test_ds.df["nodule_key"].to_numpy()[idxs])
        preds.to_csv(run_dir / "test_preds.csv", index=False)
        print(f"{run_dir.name}: {len(preds)} test nodules -> {run_dir / 'test_preds.csv'}", flush=True)

    print(f"\ndone: {len(run_dirs)} runs, device {device}. Nothing was scored.")


if __name__ == "__main__":
    main()

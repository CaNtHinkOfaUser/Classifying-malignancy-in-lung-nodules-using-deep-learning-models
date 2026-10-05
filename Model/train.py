# Written by Claude for Ishaan

"""Step D. Train one classifier run: one label type, one cross-validation fold.

Run from the project folder, with the Expansion drive connected:
    .venv/bin/python Model/train.py --labels hard --fold 1
    .venv/bin/python Model/train.py --labels soft --fold 1 --epochs 1     # quick test

--fold k (1-4) is the cross-validation fold this run is checked on: it trains
on the other three of folds 1-4 and keeps the epoch with the lowest loss on
fold k. Fold 0 is the locked test set; this script never loads it. Scoring it
is predict_test.py's job, once, at the very end. Hard and soft run k use the
same seed (k), so they start from identical weights and see the same batches.

Writes runs/cls/<labels>_fold<k>/:
    config.json     the settings of this run
    log.csv         epoch, train_loss, val_loss, val_bal_acc, seconds
    best.pt         weights from the epoch with the lowest validation loss
    val_preds.csv   nodule_key + softmax probabilities p1..p5 on the validation fold
    results.png, confusion_matrix.png, reliability.png, val_metrics.json
                    figures and numbers for the validation fold (Model/plots.py)

--overfit 32 is the "can it memorise" check from the plan: train on 32 train
nodules with augmentation and dropout off, and score those same 32. The hard
loss should fall below 0.05. The soft loss can't go below the entropy of those
32 targets, so it should settle near that number instead of 0.
"""

import argparse
import json
import random
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset import NoduleDataset
from net import NoduleNet3D

PROJECT = Path(__file__).resolve().parent.parent


def parse_args():
    p = argparse.ArgumentParser(description="Train one classifier run.")
    p.add_argument("--labels", choices=["hard", "soft"], required=True)
    p.add_argument("--fold", type=int, choices=[1, 2, 3, 4], required=True,
                   help="the cross-validation fold to validate on (fold 0 is the locked test set)")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=15, help="stop after this many epochs without a better val loss")
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--overfit", type=int, default=0, help="memorise this many train nodules (a check, not a real run)")
    p.add_argument("--out", type=Path, default=PROJECT / "runs" / "cls")
    return p.parse_args()


def run_epoch(model, loader, loss_fn, device, optimizer=None):
    """One pass over `loader`. Trains when given an optimizer, otherwise only scores.

    Returns (mean loss, softmax probabilities, dataset indices). The
    probabilities and indices are only collected when scoring.
    """
    training = optimizer is not None
    model.train(training)                  # dropout and BatchNorm behave differently in each mode
    total, count, probs, idxs = 0.0, 0, [], []
    with torch.set_grad_enabled(training):
        for x, target, idx in loader:
            x, target = x.to(device), target.to(device)
            out = model(x)
            loss = loss_fn(out, target)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            else:
                probs.append(torch.softmax(out, dim=1).cpu())
                idxs.append(idx)
            total += loss.item() * len(x)
            count += len(x)
    if training:
        return total / count, None, None
    return total / count, torch.cat(probs).numpy(), torch.cat(idxs).numpy()


def main():
    args = parse_args()
    seed = args.fold
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    name = f"{args.labels}_fold{args.fold}" + (f"_overfit{args.overfit}" if args.overfit else "")
    out_dir = args.out / name
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- data ----
    t0 = time.time()
    train_ds = NoduleDataset(args.fold, "train", args.labels, seed=seed)
    if args.overfit:
        train_ds.augment = False
        train_set = val_set = Subset(train_ds, range(args.overfit))
        val_ds = train_ds
    else:
        val_ds = NoduleDataset(args.fold, "val", args.labels, seed=seed)
        train_set, val_set = train_ds, val_ds
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_set, batch_size=args.batch, shuffle=True, num_workers=0, generator=generator)
    val_loader = DataLoader(val_set, batch_size=args.batch, shuffle=False, num_workers=0)

    # ---- model, loss, optimiser ----
    model = NoduleNet3D().to(device)
    if args.overfit:
        model.dropout.p = 0.0
    loss_fn = nn.CrossEntropyLoss()        # takes a class index (hard) or 5 probabilities (soft) unchanged
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    config = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    config.update(seed=seed, device=str(device), train_nodules=len(train_set), val_nodules=len(val_set),
                  parameters=sum(p.numel() for p in model.parameters()))
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))
    print(f"{name}: {len(train_set)} train / {len(val_set)} val nodules, "
          f"cubes loaded in {time.time() - t0:.0f}s, {config['parameters']:,} parameters, device {device}", flush=True)

    # ---- train ----
    best_loss, best_epoch, log = float("inf"), 0, []
    for epoch in range(1, args.epochs + 1):
        t = time.time()
        train_loss, _, _ = run_epoch(model, train_loader, loss_fn, device, optimizer)
        val_loss, probs, idxs = run_epoch(model, val_loader, loss_fn, device)
        hard = val_ds.hard[idxs]
        pred = probs.argmax(axis=1) + 1
        with warnings.catch_warnings():    # a 32-nodule overfit set may lack a rating
            warnings.simplefilter("ignore")
            bal = balanced_accuracy_score(hard, pred)
        row = {"epoch": epoch, "train_loss": round(train_loss, 5), "val_loss": round(val_loss, 5),
               "val_bal_acc": round(float(bal), 4), "seconds": round(time.time() - t, 1)}
        log.append(row)
        pd.DataFrame(log).to_csv(out_dir / "log.csv", index=False)

        improved = val_loss < best_loss
        if improved:
            best_loss, best_epoch = val_loss, epoch
            torch.save(model.state_dict(), out_dir / "best.pt")
        if improved or epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            print(f"epoch {epoch:3d}  train {train_loss:.4f}  val {val_loss:.4f}  "
                  f"bal {row['val_bal_acc']:.3f}  {row['seconds']:.0f}s"
                  + ("  *best" if improved else ""), flush=True)
        if epoch - best_epoch >= args.patience:
            print(f"no better val loss for {args.patience} epochs: stopping", flush=True)
            break

    # ---- predictions of the best epoch on the validation fold ----
    model.load_state_dict(torch.load(out_dir / "best.pt", map_location=device))
    _, probs, idxs = run_epoch(model, val_loader, loss_fn, device)
    preds = pd.DataFrame(probs, columns=[f"p{k}" for k in range(1, 6)])
    preds.insert(0, "nodule_key", val_ds.df["nodule_key"].to_numpy()[idxs])
    preds.to_csv(out_dir / "val_preds.csv", index=False)
    print(f"best epoch {best_epoch}, val loss {best_loss:.4f} -> {out_dir}", flush=True)

    # ---- figures and the results table, like YOLO's results.png ----
    try:
        import matplotlib
        matplotlib.use("Agg")              # save files only, never open windows
        from plots import report_run
        report_run(out_dir)
    except Exception as e:                 # a figure problem must not cost a finished run
        print(f"figures failed ({e!r}); the run itself is saved. Retry: plots.report_run('{out_dir}')")


if __name__ == "__main__":
    main()

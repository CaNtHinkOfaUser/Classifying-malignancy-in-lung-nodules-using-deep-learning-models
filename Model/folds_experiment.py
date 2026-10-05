# Written by Claude for Ishaan

"""Mini experiment: does cross-validation (folds) change anything compared with
the initial plan (one train / val / test split)?

Fold 0 is the locked test set, so this experiment never loads it. Everything
happens inside folds 1-4, the 1,508 nodules we're allowed to look at.

Part A, the luck of the split. Each cross-validation run is one possible version
of the initial plan: one training set, one set to score on. The 8 runs of the
real plan (hard and soft x folds 1-4, made by Model/train.py into runs/cls/)
are four such versions side by side. If they disagree, a single split could
have told either story.

Part B, the cost of folds. The initial plan trains each model on 1,305
nodules; cross-validation trains on 1,131, 13% fewer. Fold 1 is cut in half by
patient (balanced like the folds). Half A is the shared scoring set, and two
training sizes are compared on it:
    folds size          folds 2-4                    1,131 nodules
    initial-plan size   folds 2-4 + half B of fold 1  ~1,320 nodules
Hard and soft labels, two seeds each: the gap between seeds shows how big
training luck alone is, so the size difference can be judged against it.
Network, augmentation, optimiser and early stopping are train.py's. Half A
picks the best epoch and is scored, the same for both sizes.

Usage, from the project folder with the Expansion drive connected:
    .venv/bin/python Model/folds_experiment.py partB      # 8 runs, about 1.5 h
    .venv/bin/python Model/folds_experiment.py report     # tables + figures
Part A reads the 8 runs that Model/train.py writes to runs/cls/.
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset import SOFT_COLS, NoduleDataset   # noqa: E402  (dataset.py also puts pipeline/ on the path)
from net import NoduleNet3D                    # noqa: E402
from train import run_epoch                    # noqa: E402
import metrics as M                            # noqa: E402
from ct import NODULES, PATCH_DIR, TEST_FOLD   # noqa: E402

PROJECT = Path(__file__).resolve().parent.parent
RUNS = PROJECT / "runs"
OUT = RUNS / "folds_experiment"
RESULTS = PROJECT / "results"
EVAL_FOLD, OTHER_FOLDS = 1, (2, 3, 4)
SEEDS = (1, 2)
SIZES = {"folds": "folds size", "plan": "initial-plan size"}
KEYS = ["balanced_accuracy", "auc", "ce_vs_readers", "ece"]
NAMES = {"balanced_accuracy": "Balanced accuracy", "auc": "AUC, suspicious vs not",
         "ce_vs_readers": "CE vs readers", "ece": "ECE"}
HIGHER_IS_BETTER = {"balanced_accuracy": True, "auc": True, "ce_vs_readers": False, "ece": False}
PROBS = [f"p{k}" for k in range(1, 6)]


class KeyedDataset(NoduleDataset):
    """NoduleDataset over chosen rows instead of a whole fold.

    Cropping, augmentation and normalisation are NoduleDataset's own
    (__getitem__ is inherited), so the model sees exactly what train.py feeds it.
    """

    def __init__(self, table, cubes, labels, train, seed):
        self.df = table.reset_index(drop=True)
        self.cubes = cubes
        self.hard = self.df["hard"].to_numpy()
        self.soft = self.df[SOFT_COLS].to_numpy()
        self.labels = labels
        self.augment = train
        self.rng = np.random.default_rng(seed)


def halves(nl):
    """Fold 1's patients in two halves, balanced the way the folds are.

    Within each stratum (a patient's highest rating), patients are shuffled,
    largest first, and each goes to the half with fewer nodules of that stratum
    so far (ties: fewer nodules overall). Returns "A" or "B" per fold-1 row.
    """
    f1 = nl[nl["fold"] == EVAL_FOLD]
    pat = f1.groupby("patient_id").agg(n=("hard", "size"), top=("hard", "max")).reset_index()
    rng = np.random.default_rng(42)
    half, total = {}, {"A": 0, "B": 0}
    for _, g in pat.groupby("top"):
        g = g.iloc[rng.permutation(len(g))].sort_values("n", ascending=False, kind="stable")
        within = {"A": 0, "B": 0}
        for p, n in zip(g["patient_id"], g["n"]):
            h = min("AB", key=lambda k: (within[k], total[k]))
            half[p] = h
            within[h] += n
            total[h] += n
    return f1["patient_id"].map(half)


def part_b_sets():
    """The rows each size trains on, and half A (where both are scored)."""
    nl = pd.read_csv(NODULES)
    nl = nl[nl["fold"] != TEST_FOLD].reset_index(drop=True)          # fold 0 stays sealed
    h = halves(nl)
    half_a = nl.loc[h.index[h == "A"]]
    half_b = nl.loc[h.index[h == "B"]]
    rest = nl[nl["fold"].isin(OTHER_FOLDS)]
    return nl, {"folds": rest, "plan": pd.concat([rest, half_b])}, half_a


def fit(train_rows, val_rows, cubes, labels, seed, out_dir, epochs=100, patience=15, batch=32):
    """train.py's loop: AdamW 1e-3, weight decay 1e-4, batch 32, keep the lowest val loss,
    stop after 15 epochs without one. Same seeding scheme as train.py."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    train_ds = KeyedDataset(train_rows, cubes[train_rows.index], labels, True, seed)
    val_ds = KeyedDataset(val_rows, cubes[val_rows.index], labels, False, seed)
    train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True, num_workers=0,
                              generator=torch.Generator().manual_seed(seed))
    val_loader = DataLoader(val_ds, batch_size=batch, shuffle=False, num_workers=0)

    model = NoduleNet3D().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best_loss, best_epoch, log = float("inf"), 0, []
    out_dir.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, epochs + 1):
        t = time.time()
        train_loss, _, _ = run_epoch(model, train_loader, loss_fn, device, optimizer)
        val_loss, _, _ = run_epoch(model, val_loader, loss_fn, device)
        log.append({"epoch": epoch, "train_loss": round(train_loss, 5), "val_loss": round(val_loss, 5),
                    "seconds": round(time.time() - t, 1)})
        if val_loss < best_loss:
            best_loss, best_epoch = val_loss, epoch
            torch.save(model.state_dict(), out_dir / "best.pt")
        if epoch - best_epoch >= patience:
            break
    pd.DataFrame(log).to_csv(out_dir / "log.csv", index=False)

    model.load_state_dict(torch.load(out_dir / "best.pt", map_location=device))
    _, probs, idxs = run_epoch(model, val_loader, loss_fn, device)
    preds = pd.DataFrame(probs, columns=PROBS)
    preds.insert(0, "nodule_key", val_ds.df["nodule_key"].to_numpy()[idxs])
    preds.to_csv(out_dir / "val_preds.csv", index=False)
    (out_dir / "run.json").write_text(json.dumps(
        {"labels": labels, "seed": seed, "train_nodules": len(train_ds), "scored_nodules": len(val_ds),
         "best_epoch": best_epoch, "epochs_run": len(log)}, indent=2))
    return best_epoch, len(log)


def part_b():
    nl, train_sets, half_a = part_b_sets()
    print(f"half A (scored): {len(half_a)} nodules, {half_a['patient_id'].nunique()} patients")
    for size, rows in train_sets.items():
        print(f"{SIZES[size]}: trains on {len(rows)} nodules")
    t0 = time.time()
    cubes = np.stack([np.load(PATCH_DIR / f) for f in nl["patch_file"]])   # folds 1-4, loaded once
    print(f"{len(cubes)} cubes loaded in {time.time() - t0:.0f}s", flush=True)
    for labels in ("hard", "soft"):
        for seed in SEEDS:
            for size, rows in train_sets.items():
                out_dir = OUT / f"{size}_{labels}_seed{seed}"
                if (out_dir / "val_preds.csv").exists():
                    print(f"{out_dir.name}: done already", flush=True)
                    continue
                t = time.time()
                best, ran = fit(rows, half_a, cubes, labels, seed, out_dir)
                print(f"{out_dir.name}: best epoch {best} of {ran}, {(time.time() - t) / 60:.1f} min", flush=True)


def _score(preds_file, nl):
    d = pd.read_csv(preds_file).merge(nl[["nodule_key", "hard", *SOFT_COLS]], on="nodule_key", validate="one_to_one")
    return M.scores(d[PROBS].to_numpy(), d["hard"].to_numpy().astype(int), d[SOFT_COLS].to_numpy())


def report():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from plots import INK_2, MUTED, TRAIN, VAL, _legend, _style
    HARD, SOFT = TRAIN, VAL
    nl = pd.read_csv(NODULES)
    RESULTS.mkdir(exist_ok=True)

    # ---- Part A: the 8 cross-validation runs, one row per fold ----
    files = {(lab, k): RUNS / "cls" / f"{lab}_fold{k}" / "val_metrics.json" for lab in ("hard", "soft") for k in (1, 2, 3, 4)}
    if all(f.exists() for f in files.values()):
        a = pd.DataFrame([{"labels": lab, "fold": k, **{m: json.loads(f.read_text())[m] for m in KEYS}}
                          for (lab, k), f in files.items()])
        size = pd.read_csv(RESULTS / "baselines_val_per_run.csv").set_index("val_fold")
        a.to_csv(RESULTS / "folds_experiment_partA.csv", index=False)
        print("PART A: each fold = one version of the initial plan (scored on that fold, 377 nodules)")
        for m in KEYS:
            w = a.pivot(index="fold", columns="labels", values=m)
            w["soft - hard"] = w["soft"] - w["hard"]
            w["size only"] = size[m]
            better = (w["soft - hard"] > 0) if HIGHER_IS_BETTER[m] else (w["soft - hard"] < 0)
            print(f"\n  {NAMES[m]} ({'higher' if HIGHER_IS_BETTER[m] else 'lower'} is better); "
                  f"soft better on {int(better.sum())} of 4 folds")
            print("  " + w.round(3).to_string().replace("\n", "\n  "))
            print(f"  mean ± sd    hard {w['hard'].mean():.3f} ± {w['hard'].std():.3f}   "
                  f"soft {w['soft'].mean():.3f} ± {w['soft'].std():.3f}   "
                  f"soft - hard {w['soft - hard'].mean():+.3f} ± {w['soft - hard'].std():.3f}   "
                  f"range of one split: {w['soft - hard'].min():+.3f} to {w['soft - hard'].max():+.3f}")

        fig, axes = plt.subplots(1, 4, figsize=(15, 3.9))
        for ax, m in zip(axes, KEYS):
            w = a.pivot(index="fold", columns="labels", values=m)
            x = np.arange(1, 5)
            for k in x:
                ax.plot([k - 0.12, k + 0.12], [w.loc[k, "hard"], w.loc[k, "soft"]], color=MUTED, linewidth=1, zorder=1)
            ax.scatter(x - 0.12, w["hard"], s=46, color=HARD, zorder=3, label="hard labels",
                       edgecolor="white", linewidth=1)
            ax.scatter(x + 0.12, w["soft"], s=46, color=SOFT, zorder=3, label="soft labels",
                       edgecolor="white", linewidth=1)
            ax.scatter(x, size[m].loc[x], marker="_", s=260, color=MUTED, linewidth=2, zorder=2, label="size only")
            _style(ax, f"{NAMES[m]}\n({'higher' if HIGHER_IS_BETTER[m] else 'lower'} is better)", pad=24)
            ax.set_xticks(x, [f"fold {k}" for k in x])
            ax.set_xlim(0.5, 4.5)
            _legend(ax)
        fig.suptitle("Part A: four single splits, side by side (validation folds 1-4)", x=0.01, ha="left",
                     fontsize=12, color=INK_2)
        fig.tight_layout()
        fig.savefig(RESULTS / "folds_experiment_partA.png", dpi=150, facecolor="white")
        plt.close(fig)
    else:
        missing = [f"{lab}_fold{k}" for (lab, k), f in files.items() if not f.exists()]
        print(f"PART A: waiting for runs/cls/{', '.join(missing)}")

    # ---- Part B: initial-plan size vs folds size, scored on the same half of fold 1 ----
    rows = []
    for size in SIZES:
        for labels in ("hard", "soft"):
            for seed in SEEDS:
                f = OUT / f"{size}_{labels}_seed{seed}" / "val_preds.csv"
                if f.exists():
                    rows.append({"size": size, "labels": labels, "seed": seed, **_score(f, nl)})
    if len(rows) < len(SIZES) * 2 * len(SEEDS):
        print(f"\nPART B: {len(rows)} of {len(SIZES) * 2 * len(SEEDS)} runs done")
        return
    b = pd.DataFrame(rows)
    b.to_csv(RESULTS / "folds_experiment_partB.csv", index=False)
    n_scored = json.loads((OUT / "folds_hard_seed1" / "run.json").read_text())["scored_nodules"]
    n_train = {s: json.loads((OUT / f"{s}_hard_seed1" / "run.json").read_text())["train_nodules"] for s in SIZES}
    print(f"\nPART B: scored on half A of fold 1 ({n_scored} nodules); mean of seeds {SEEDS}")
    for labels in ("hard", "soft"):
        g = b[b["labels"] == labels]
        mean = g.groupby("size")[KEYS].mean()
        gap = g.groupby("size")[KEYS].agg(lambda s: s.max() - s.min()).mean()
        t = pd.DataFrame({f"{labels}, folds size ({n_train['folds']})": mean.loc["folds"],
                          f"{labels}, initial-plan size ({n_train['plan']})": mean.loc["plan"],
                          "  more data minus less": mean.loc["plan"] - mean.loc["folds"],
                          "  gap between the 2 seeds": gap}).T
        print(t.round(3).to_string())

    fig, axes = plt.subplots(1, 4, figsize=(15, 3.9))
    for ax, m in zip(axes, KEYS):
        for j, (labels, colour) in enumerate((("hard", HARD), ("soft", SOFT))):
            for i, size in enumerate(SIZES):
                v = b[(b["labels"] == labels) & (b["size"] == size)][m].to_numpy()
                xpos = j * 2.6 + i
                ax.scatter([xpos] * len(v), v, s=40, color=colour, edgecolor="white", linewidth=1, zorder=3,
                           alpha=0.55 if size == "folds" else 1.0,
                           label=(f"{labels} labels" if i == 1 else None))
                ax.plot([xpos - 0.28, xpos + 0.28], [v.mean()] * 2, color=colour, linewidth=2.2, zorder=2)
        ax.set_xticks([0, 1, 2.6, 3.6], [f"{n_train['folds']}\nfolds", f"{n_train['plan']}\nplan"] * 2)
        ax.set_xlim(-0.6, 4.2)
        _style(ax, f"{NAMES[m]}\n({'higher' if HIGHER_IS_BETTER[m] else 'lower'} is better)", pad=24)
        _legend(ax)
    fig.suptitle(f"Part B: training nodules, folds vs initial plan; dots = seeds, line = mean "
                 f"(scored on {n_scored} fold-1 nodules)", x=0.01, ha="left", fontsize=12, color=INK_2)
    fig.tight_layout()
    fig.savefig(RESULTS / "folds_experiment_partB.png", dpi=150, facecolor="white")
    plt.close(fig)
    print(f"\nfigures: {RESULTS / 'folds_experiment_partA.png'}, {RESULTS / 'folds_experiment_partB.png'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Folds vs the initial single split.")
    p.add_argument("step", choices=["partB", "report"])
    if p.parse_args().step == "partB":
        part_b()
    else:
        report()

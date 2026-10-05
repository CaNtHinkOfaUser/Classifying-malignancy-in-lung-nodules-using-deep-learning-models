# Written by Claude for Ishaan

"""Figures and a results table for one classifier run, like YOLO's results.png.

train.py calls this automatically when a run finishes. From a notebook:
    import sys; sys.path.insert(0, "Model")
    from plots import report_run
    report_run("runs/cls/hard_fold0", show=True)

Writes into the run folder:
    results.png           loss and validation balanced accuracy per epoch, best epoch marked
    confusion_matrix.png  median reader rating vs predicted rating
    reliability.png       how often the model was right at each confidence level
    val_metrics.json      every number in the printed table
All numbers are for the run's validation fold at its best epoch. "Size only" is
the diameter-only baseline on the same fold (from pipeline/baselines.py).
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MaxNLocator

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT / "pipeline"))
import metrics as M                       # noqa: E402
from ct import NODULES                    # noqa: E402

# Chart colours. Blue = training, orange = validation (checked for colour-blind safety).
TRAIN, VAL = "#2a78d6", "#eb6834"
INK, INK_2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
BLUES = LinearSegmentedColormap.from_list("blues", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
SOFT = [f"soft_{k}" for k in range(1, 6)]
PROBS = [f"p{k}" for k in range(1, 6)]


def _style(ax, title, pad=10):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=11, color=INK, pad=pad)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=GRID, labelcolor=INK_2, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


DASHED, DOTTED = (0, (5, 3)), (0, (1, 2))


def _ref(ax, y, label, style):
    """A grey reference line, named in the panel's legend (dashed: size only; dotted: floors)."""
    ax.axhline(y, color=MUTED, linewidth=1.2, linestyle=style, zorder=1, label=label)


def _legend(ax):
    """Legend in the band between the panel title and the plot, so it never covers data."""
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK_2, ncol=3, handlelength=2.2, columnspacing=1.2,
              loc="lower left", bbox_to_anchor=(0, 1.0), borderaxespad=0.2)


def _best_epoch(ax, epoch, xmax, label=False):
    ax.axvline(epoch, color=MUTED, linewidth=1, linestyle=(0, (3, 2, 1, 2)), zorder=1)
    if label:                             # label on whichever side of the line has more room
        right = epoch < 0.6 * xmax
        ax.annotate(f"best epoch {epoch}", xy=(epoch, 1), xycoords=("data", "axes fraction"),
                    xytext=(4 if right else -4, -12), textcoords="offset points",
                    ha="left" if right else "right", fontsize=8.5, color=INK_2)


def _line(ax, x, y, color, label=None):
    many = len(x) > 30                    # dots only while there are few epochs
    ax.plot(x, y, color=color, linewidth=1.8, solid_capstyle="round", label=label,
            marker=None if many else "o", markersize=4.5, markeredgecolor=SURFACE, markeredgewidth=1)


def _results_png(log, best, floors, size, title, path):
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3), facecolor=SURFACE)
    ep = log["epoch"]
    xmax = max(int(ep.max()), 2)

    ax = axes[0]
    _style(ax, "Loss", pad=26)
    _line(ax, ep, log["train_loss"], TRAIN, "train")
    _line(ax, ep, log["val_loss"], VAL, "validation")
    if floors.get("loss") is not None:
        _ref(ax, floors["loss"], f"lowest possible ({floors['loss']:.3f})", DOTTED)
    _best_epoch(ax, best, xmax, label=True)
    ax.set_ylim(0, max(log["train_loss"].max(), log["val_loss"].max()) * 1.12)
    _legend(ax)

    ax = axes[1]
    _style(ax, "Validation balanced accuracy", pad=26)
    _line(ax, ep, log["val_bal_acc"], VAL, "validation")
    if "balanced_accuracy" in size:
        _ref(ax, size["balanced_accuracy"], f"size only ({size['balanced_accuracy']:.3f})", DASHED)
    _ref(ax, 0.2, "chance (0.200)", DOTTED)
    _best_epoch(ax, best, xmax)
    ax.set_ylim(0, 1)
    _legend(ax)

    for ax in axes:
        ax.set_xlim(0.5, xmax + 0.5)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))    # whole epochs only
        ax.set_xlabel("epoch", fontsize=9, color=INK_2)
    fig.suptitle(title, x=0.01, ha="left", fontsize=12, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def _confusion_png(hard, pred, title, path):
    counts = np.zeros((5, 5), dtype=int)
    for t, p in zip(hard, pred):
        counts[t - 1, p - 1] += 1
    share = counts / np.maximum(counts.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(5.4, 4.6), facecolor=SURFACE)
    im = ax.imshow(share, cmap=BLUES, vmin=0, vmax=1)
    for i in range(5):
        for j in range(5):
            ax.text(j, i, str(counts[i, j]), ha="center", va="center", fontsize=10,
                    color="white" if share[i, j] > 0.55 else INK)
    ax.set_xticks(range(5), [1, 2, 3, 4, 5])
    ax.set_yticks(range(5), [f"{k}  (n={counts[k - 1].sum()})" for k in range(1, 6)])
    ax.set_xlabel("predicted rating", fontsize=9, color=INK_2)
    ax.set_ylabel("median reader rating", fontsize=9, color=INK_2)
    ax.tick_params(length=0, labelcolor=INK_2, labelsize=9)
    for spine in ax.spines.values():
        spine.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("share of the row", fontsize=9, color=INK_2)
    cb.ax.tick_params(labelsize=8, labelcolor=INK_2, color=GRID)
    cb.outline.set_visible(False)
    ax.set_title(title, loc="left", fontsize=11, color=INK, pad=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def _reliability_png(probs, hard, ece, title, path, min_n=10):
    conf = probs.max(axis=1)
    correct = (probs.argmax(axis=1) + 1) == hard
    edges = np.linspace(0, 1, 11)
    fig, ax = plt.subplots(figsize=(5.6, 4.8), facecolor=SURFACE)
    _style(ax, f"{title}  (ECE {ece:.3f})", pad=22)
    ax.annotate(f"dashed: perfectly calibrated · numbers: nodules per bin · pale: fewer than {min_n}",
                xy=(0, 1), xycoords="axes fraction", xytext=(0, 5), textcoords="offset points",
                va="bottom", fontsize=8, color=INK_2)
    ax.plot([0, 1], [0, 1], color=MUTED, linewidth=1.2, linestyle=DASHED, zorder=1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if not m.any():
            continue
        acc = correct[m].mean()
        ax.bar((lo + hi) / 2, acc, width=0.07, color=VAL, alpha=1.0 if m.sum() >= min_n else 0.35, zorder=2)
        ax.annotate(f"{int(m.sum())}", xy=((lo + hi) / 2, acc), xytext=(0, 3), textcoords="offset points",
                    ha="center", fontsize=8, color=INK_2)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.1)
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.set_xlabel("model's confidence (its top probability)", fontsize=9, color=INK_2)
    ax.set_ylabel("share it got right", fontsize=9, color=INK_2)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def report_run(run_dir, show=False):
    """Print the run's validation results and save its three figures. Returns the numbers."""
    run_dir = Path(run_dir)
    cfg = json.loads((run_dir / "config.json").read_text())
    log = pd.read_csv(run_dir / "log.csv")
    preds = pd.read_csv(run_dir / "val_preds.csv")
    labels = pd.read_csv(NODULES)[["nodule_key", "hard", *SOFT]]
    d = preds.merge(labels, on="nodule_key", how="left", validate="one_to_one")
    assert d["hard"].notna().all(), "a prediction has no label"

    probs = d[PROBS].to_numpy()
    hard = d["hard"].to_numpy().astype(int)
    soft = d[SOFT].to_numpy()
    pred = probs.argmax(axis=1) + 1
    m = M.scores(probs, hard, soft)

    entropy = float(-(soft * np.log(np.clip(soft, 1e-12, 1))).sum(axis=1).mean())
    floors = {"loss": entropy if cfg["labels"] == "soft" else None}
    size = {}
    per_run = PROJECT / "results" / "baselines_val_per_run.csv"
    if per_run.exists() and not cfg.get("overfit"):
        b = pd.read_csv(per_run).set_index("val_fold")
        if cfg["fold"] in b.index:
            size = b.loc[cfg["fold"]].to_dict()

    best = int(log.loc[log["val_loss"].idxmin(), "epoch"])
    where = "the same 32 train nodules" if cfg.get("overfit") else f"validation fold {cfg['fold']}"
    name = run_dir.name
    title = f"{name}  ·  {where}  ·  best epoch {best} of {int(log['epoch'].max())}  ·  {len(d)} nodules"

    _results_png(log, best, floors, size, title, run_dir / "results.png")
    _confusion_png(hard, pred, f"{name}: confusion matrix", run_dir / "confusion_matrix.png")
    _reliability_png(probs, hard, m["ece"], f"{name}: reliability", run_dir / "reliability.png")

    # ---- the printed table ----
    rows = [("balanced accuracy", "balanced_accuracy", "0.200  chance"),
            ("AUC, suspicious vs not", "auc", "0.500  chance"),
            ("CE vs readers", "ce_vs_readers", f"{entropy:.3f}  lowest possible"),
            ("ECE", "ece", "0.000  perfectly calibrated")]
    print(f"\n{title}")
    print(f"  {'':24s}{'this run':>10s}{'size only':>11s}   reference")
    for label, key, ref in rows:
        s = f"{size[key]:11.3f}" if key in size else f"{'—':>11s}"
        print(f"  {label:24s}{m[key]:10.3f}{s}   {ref}")
    print(f"  figures: {run_dir / 'results.png'}, confusion_matrix.png, reliability.png")

    out = {k: float(v) for k, v in m.items()} | {"best_epoch": best, "nodules": len(d),
            "lowest_possible_ce": entropy,
            "size_only": {k: float(v) for k, v in size.items()}}
    (run_dir / "val_metrics.json").write_text(json.dumps(out, indent=2))

    if show:
        from IPython.display import Image, display
        for f in ("results.png", "confusion_matrix.png", "reliability.png"):
            display(Image(filename=str(run_dir / f)))
    return out

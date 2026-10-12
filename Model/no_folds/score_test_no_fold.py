"""Score the TEST split for the no-folds experiment. Run this once, after every decision is fixed.

Reads each run's test_preds.csv (from predict_test_no_fold.py) and val_preds.csv,
plus data/nodules_labels.csv, and writes into --out:
    per_run.csv                 every metric for every run on the test set
    summary.csv                 mean and standard deviation over the 3 seeds, per model
    b_minus_a.csv               B (soft) - A (hard) per metric, with a 95% bootstrap interval
    confidence_by_spread.csv    mean confidence per level of reader disagreement (+ readers' agreement)
    confidence_by_spread.png    the same as a figure, built exactly as on validation
    confusion_<run>.png         confusion matrix of each run on the test set

Decisions built in (change them HERE, before running, never after seeing results):
  * Cut-off for sensitivity / specificity: each run's cut-off is chosen on its OWN
    validation predictions (Youden's J) and applied unchanged to its test predictions.
  * Seeds: every metric is reported per run and as mean +/- SD over seeds 1-3.
    The B - A intervals pass all three seeds of each model to metrics.bootstrap_diff,
    which averages the seeds inside every resample.
  * Bootstrap: 2,000 resamples of whole PATIENTS (a patient's nodules aren't independent).
    If the interval contains 0, no difference can be claimed.
  * "Suspicious" = hard label >= 4 (rounded median; tied medians round up).

Run from the project folder:
    .venv/bin/python Model/no_folds/score_test_no_fold.py --runs-dir runs/cls_no_fold
On Colab:
    !python Model/no_folds/score_test_no_fold.py --runs-dir "{RUNS_DIR}" --out "{DRIVE_DIR}/results_test_no_fold"
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                  # save files only, never open windows
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))                 # Model/: plots.py
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "pipeline"))     # metrics.py, ct.py
import metrics as M                    # noqa: E402
from ct import NODULES                 # noqa: E402
from plots import _confusion_png       # noqa: E402

PROJECT = Path(__file__).resolve().parents[2]
SEEDS = (1, 2, 3)
MODELS = {"hard": "A (hard)", "soft": "B (soft)"}
PROBS = [f"p{k}" for k in range(1, 6)]
SOFT = [f"soft_{k}" for k in range(1, 6)]
SPREAD_GROUPS = [0, 1, 2, 3]           # 3 means "3 or more", as on validation
SPREAD_NAMES = ["0 (agreed)", "1", "2", "3+ (disagreed most)"]

# metrics with a 95% bootstrap interval for B - A. Each takes (probs, hard, soft).
# Sensitivity / specificity are left out: each seed has its own cut-off.
BOOTSTRAP = {
    "balanced_accuracy": lambda p, h, s: M.balanced_accuracy_score(h, p.argmax(axis=1) + 1),
    "auc": lambda p, h, s: M.roc_auc_score(M.suspicious(h), M.p_suspicious(p)),
    "ce_vs_readers": lambda p, h, s: -(s * np.log(np.clip(p, M.EPS, 1))).sum(axis=1).mean(),
    "ece": lambda p, h, s: M.ece(p, h),
    "quadratic_kappa": lambda p, h, s: M.quadratic_kappa(p, h),
}


def parse_args():
    p = argparse.ArgumentParser(description="Score the test split for the no-folds experiment.")
    p.add_argument("--runs-dir", type=Path, default=PROJECT / "runs" / "cls_no_fold",
                   help="the folder holding hard_seed1 ... soft_seed3")
    p.add_argument("--out", type=Path, default=PROJECT / "results" / "test_no_fold",
                   help="where to write the tables and figures")
    p.add_argument("--resamples", type=int, default=2000)
    return p.parse_args()


def load_preds(path, keys):
    """A run's probabilities, in the order of `keys`."""
    preds = pd.read_csv(path).set_index("nodule_key")
    missing = set(keys) - set(preds.index)
    if missing:
        sys.exit(f"{path} is missing {len(missing)} nodules, e.g. {sorted(missing)[0]}")
    return preds.loc[keys, PROBS].to_numpy()


def main():
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    labels = pd.read_csv(NODULES)
    test = labels[labels["split"] == "test"].sort_values("nodule_key").reset_index(drop=True)
    val = labels[labels["split"] == "val"].sort_values("nodule_key").reset_index(drop=True)
    hard, soft = test["hard"].to_numpy(), test[SOFT].to_numpy()
    y = M.suspicious(hard)
    print(f"test: {len(test)} nodules from {test['patient_id'].nunique()} patients, "
          f"{int(y.sum())} suspicious\n")

    # ---- every run on the test set ----
    rows, test_probs = [], {}
    for labels_type, model in MODELS.items():
        for seed in SEEDS:
            run = f"{labels_type}_seed{seed}"
            run_dir = args.runs_dir / run
            for f in ("test_preds.csv", "val_preds.csv"):
                if not (run_dir / f).exists():
                    sys.exit(f"{run_dir / f} not found: run predict_test_no_fold.py first")
            p_test = load_preds(run_dir / "test_preds.csv", test["nodule_key"])
            p_val = load_preds(run_dir / "val_preds.csv", val["nodule_key"])
            test_probs[labels_type, seed] = p_test

            # the threshold comes from VALIDATION and is applied unchanged to test
            threshold = M.youden_threshold(M.p_suspicious(p_val), M.suspicious(val["hard"].to_numpy()))
            m = M.all_metrics(p_test, hard, soft, threshold=threshold)
            m["threshold_from_val"] = m.pop("threshold")
            rows.append({"run": run, "model": model, "seed": seed} | m)

            _confusion_png(hard, p_test.argmax(axis=1) + 1, f"{run}: confusion matrix (test set)",
                           args.out / f"confusion_{run}.png")

    per_run = pd.DataFrame(rows)
    per_run.to_csv(args.out / "per_run.csv", index=False)
    metric_cols = ["balanced_accuracy", "auc", "ce_vs_readers", "ece", "quadratic_kappa",
                   "sensitivity", "specificity"]

    # ---- mean and SD over seeds ----
    summary = per_run.groupby("model")[metric_cols].agg(["mean", "std"]).round(3)
    summary.to_csv(args.out / "summary.csv")

    # ---- B - A with 95% bootstrap intervals, resampling whole patients ----
    # bootstrap_diff returns metric(first) - metric(second): soft first gives B - A
    soft_runs = np.stack([test_probs["soft", s] for s in SEEDS])
    hard_runs = np.stack([test_probs["hard", s] for s in SEEDS])
    diffs = []
    for name, fn in BOOTSTRAP.items():
        mean, (lo, hi) = M.bootstrap_diff(fn, soft_runs, hard_runs, hard, soft, n=args.resamples,
                                          seed=0, groups=test["patient_id"].to_numpy())
        diffs.append({"metric": name, "b_minus_a": round(mean, 4), "ci_low": lo, "ci_high": hi,
                      "interval_excludes_0": bool(lo > 0 or hi < 0)})
        print(f"  bootstrap {name} done", flush=True)
    b_minus_a = pd.DataFrame(diffs)
    b_minus_a.to_csv(args.out / "b_minus_a.csv", index=False)

    # ---- confidence by reader disagreement, as on validation ----
    spread = np.minimum(test["spread"].to_numpy(), 3)
    readers = test[SOFT].max(axis=1).groupby(spread).mean()
    cbs_rows = []
    for labels_type, model in MODELS.items():
        per_seed = [M.confidence_by_spread(test_probs[labels_type, s], spread) for s in SEEDS]
        for g, name in zip(SPREAD_GROUPS, SPREAD_NAMES):
            conf = [r[g][0] for r in per_seed]
            cbs_rows.append({"model": model, "spread": name, "nodules": per_seed[0][g][1],
                             "mean_confidence": round(float(np.mean(conf)), 3),
                             "lowest_seed": round(min(conf), 3), "highest_seed": round(max(conf), 3),
                             "readers_agreement": round(float(readers.loc[g]), 3)})
    cbs = pd.DataFrame(cbs_rows)
    cbs.to_csv(args.out / "confidence_by_spread.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for labels_type, model, colour in (("hard", "A (hard labels)", "tab:blue"),
                                       ("soft", "B (soft labels)", "tab:orange")):
        conf = np.array([[M.confidence_by_spread(test_probs[labels_type, s], spread)[g][0]
                          for g in SPREAD_GROUPS] for s in SEEDS])
        ax.plot(SPREAD_GROUPS, conf.mean(axis=0), marker="o", color=colour, label=model)
        ax.fill_between(SPREAD_GROUPS, conf.min(axis=0), conf.max(axis=0), color=colour, alpha=0.15)
    ax.plot(SPREAD_GROUPS, readers.loc[SPREAD_GROUPS].to_numpy(), "k--", marker="s",
            label="readers' own agreement (ideal)")
    counts = cbs[cbs["model"] == MODELS["hard"]]["nodules"].tolist()
    ax.set_xticks(SPREAD_GROUPS, [f"{n}\nn = {c}" for n, c in zip(SPREAD_NAMES, counts)])
    ax.set_xlabel("reader disagreement (highest rating minus lowest)")
    ax.set_ylabel("model's confidence (its top probability)")
    ax.set_title("Model confidence by reader disagreement\n"
                 "TEST set, mean of 3 seeds (shaded: lowest to highest seed)", fontsize=10)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(args.out / "confidence_by_spread.png", dpi=150)
    plt.close(fig)

    # ---- print ----
    pd.set_option("display.width", 200)
    print("\nper run (test set):")
    print(per_run.drop(columns=["model", "seed"]).round(3).to_string(index=False))
    print("\nmean and SD over seeds:")
    print(summary.to_string())
    print("\nB (soft) - A (hard), 95% bootstrap interval over patients:")
    print(b_minus_a.to_string(index=False))
    print("\nconfidence by reader disagreement:")
    print(cbs.to_string(index=False))
    print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()

"""Score the locked TEST fold (fold 0) for the cross-validation experiment. Run this once, after every decision is fixed.

Reads each run's test_preds.csv (from predict_test_fold.py) and val_preds.csv,
plus data/nodules_labels.csv, and writes into --out:
    per_run.csv                 every metric for every run on the test fold, plus the
                                two 4-model ensembles and the size-only baseline
    summary.csv                 mean and standard deviation over the 4 folds, per model
    b_minus_a.csv               B (soft) - A (hard) per metric, with a 95% bootstrap interval
    confidence_by_spread.csv    mean confidence per level of reader disagreement (+ readers' agreement)
    confidence_by_spread.png    the same as a figure
    confusion_<run>.png         confusion matrix of each run on the test fold

Decisions built in (change them HERE, before running, never after seeing results):
  * The 4 cross-validation models of each type are reported per fold and as
    mean +/- SD. The ensemble (the average of the 4 models' probabilities, the
    "k-fold hypothesis" of Blum, Kalai & Langford, 1999) is reported as a second row.
  * Cut-off for sensitivity / specificity: each run's cut-off is chosen on its OWN
    validation fold (Youden's J) and applied unchanged to fold 0. The ensemble's
    cut-off comes from the pooled validation predictions of folds 1-4 (each nodule
    predicted by the one model that never trained on it).
  * Bootstrap: 2,000 resamples of whole PATIENTS. B - A pairs the 4 soft runs
    with the 4 hard runs (fold k with fold k) and averages them inside every
    resample. If the interval contains 0, no difference can be claimed.
  * Size only: for each fold k, a logistic regression on the standardised
    log-diameter, trained on run k's training folds, scored on fold 0.
  * "Suspicious" = hard label >= 4 (rounded median; tied medians round up).

Run from the project folder:
    .venv/bin/python Model/folds/score_test_fold.py --runs-dir runs/cls --out results/test_fold
    .venv/bin/python Model/folds/score_test_fold.py --runs-dir runs/cls_mask --out results/test_fold_mask
On Colab:
    !python Model/folds/score_test_fold.py --runs-dir "{RUNS_DIR}" --out "{DRIVE_DIR}/results_test_fold"
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                  # save files only, never open windows
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))                 # Model/: plots.py
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "pipeline"))     # metrics.py, ct.py
import metrics as M                    # noqa: E402
from ct import CV_FOLDS, NODULES, TEST_FOLD   # noqa: E402
from plots import _confusion_png       # noqa: E402

PROJECT = Path(__file__).resolve().parents[2]
MODELS = {"hard": "A (hard)", "soft": "B (soft)"}
PROBS = [f"p{k}" for k in range(1, 6)]
SOFT = [f"soft_{k}" for k in range(1, 6)]
SPREAD_GROUPS = [0, 1, 2, 3]           # 3 means "3 or more", as on validation
SPREAD_NAMES = ["0 (agreed)", "1", "2", "3+ (disagreed most)"]
METRIC_COLS = ["balanced_accuracy", "auc", "ce_vs_readers", "ece", "quadratic_kappa",
               "sensitivity", "specificity"]

# metrics with a 95% bootstrap interval for B - A. Each takes (probs, hard, soft).
# Sensitivity / specificity are left out: each fold has its own cut-off.
BOOTSTRAP = {
    "balanced_accuracy": lambda p, h, s: M.balanced_accuracy_score(h, p.argmax(axis=1) + 1),
    "auc": lambda p, h, s: M.roc_auc_score(M.suspicious(h), M.p_suspicious(p)),
    "ce_vs_readers": lambda p, h, s: -(s * np.log(np.clip(p, M.EPS, 1))).sum(axis=1).mean(),
    "ece": lambda p, h, s: M.ece(p, h),
    "quadratic_kappa": lambda p, h, s: M.quadratic_kappa(p, h),
}


def parse_args():
    p = argparse.ArgumentParser(description="Score the locked test fold for the cross-validation experiment.")
    p.add_argument("--runs-dir", type=Path, default=PROJECT / "runs" / "cls",
                   help="the folder holding hard_fold1 ... soft_fold4")
    p.add_argument("--out", type=Path, default=PROJECT / "results" / "test_fold",
                   help="where to write the tables and figures")
    p.add_argument("--resamples", type=int, default=2000)
    p.add_argument("--labels-file", type=Path, default=NODULES,
                   help="nodules_labels.csv whose `fold` column the runs were trained with "
                        "(the first folds: data/old_folds_v1/nodules_labels.csv)")
    return p.parse_args()


def load_preds(path, keys):
    """A run's probabilities, in the order of `keys`."""
    preds = pd.read_csv(path).set_index("nodule_key")
    missing = set(keys) - set(preds.index)
    if missing:
        sys.exit(f"{path} is missing {len(missing)} nodules, e.g. {sorted(missing)[0]}")
    return preds.loc[list(keys), PROBS].to_numpy()


def size_only(labels, k, test):
    """Diameter-only baseline for cross-validation run k: train on its training folds, predict fold 0 and fold k."""
    train = labels[~labels["fold"].isin([TEST_FOLD, k])]
    x = np.log(train["diameter_mm"].to_numpy())
    mu, sd = x.mean(), x.std()
    clf = LogisticRegression(max_iter=1000).fit(((x - mu) / sd)[:, None], train["hard"].to_numpy())

    def predict(rows):
        return clf.predict_proba(((np.log(rows["diameter_mm"].to_numpy()) - mu) / sd)[:, None])

    val = labels[labels["fold"] == k].sort_values("nodule_key")
    return predict(test), predict(val), val


def main():
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    labels = pd.read_csv(args.labels_file)
    test = labels[labels["fold"] == TEST_FOLD].sort_values("nodule_key").reset_index(drop=True)
    hard, soft = test["hard"].to_numpy(), test[SOFT].to_numpy()
    y = M.suspicious(hard)
    print(f"test fold {TEST_FOLD}: {len(test)} nodules from {test['patient_id'].nunique()} patients, "
          f"{int(y.sum())} suspicious\n")

    # ---- every cross-validation run on the test fold ----
    rows, test_probs, val_pool = [], {}, {lab: [] for lab in MODELS}
    for labels_type, model in MODELS.items():
        for k in CV_FOLDS:
            run = f"{labels_type}_fold{k}"
            run_dir = args.runs_dir / run
            for f in ("test_preds.csv", "val_preds.csv"):
                if not (run_dir / f).exists():
                    sys.exit(f"{run_dir / f} not found: run predict_test_fold.py first")
            val = labels[labels["fold"] == k].sort_values("nodule_key")
            p_test = load_preds(run_dir / "test_preds.csv", test["nodule_key"])
            p_val = load_preds(run_dir / "val_preds.csv", val["nodule_key"])
            test_probs[labels_type, k] = p_test
            val_pool[labels_type].append((p_val, val["hard"].to_numpy()))

            # the cut-off comes from this run's VALIDATION fold and is applied unchanged to fold 0
            threshold = M.youden_threshold(M.p_suspicious(p_val), M.suspicious(val["hard"].to_numpy()))
            m = M.all_metrics(p_test, hard, soft, threshold=threshold)
            m["threshold_from_val"] = m.pop("threshold")
            rows.append({"run": run, "model": model, "fold": k} | m)
            _confusion_png(hard, p_test.argmax(axis=1) + 1, f"{run}: confusion matrix (test fold 0)",
                           args.out / f"confusion_{run}.png")

    # ---- the 4-model ensembles: average the probabilities ----
    for labels_type, model in MODELS.items():
        p_test = np.mean([test_probs[labels_type, k] for k in CV_FOLDS], axis=0)
        p_val = np.concatenate([p for p, _ in val_pool[labels_type]])
        h_val = np.concatenate([h for _, h in val_pool[labels_type]])
        threshold = M.youden_threshold(M.p_suspicious(p_val), M.suspicious(h_val))
        m = M.all_metrics(p_test, hard, soft, threshold=threshold)
        m["threshold_from_val"] = m.pop("threshold")
        rows.append({"run": f"{labels_type}_ensemble", "model": f"{model} ensemble", "fold": "1-4"} | m)
        _confusion_png(hard, p_test.argmax(axis=1) + 1, f"{labels_type} ensemble: confusion matrix (test fold 0)",
                       args.out / f"confusion_{labels_type}_ensemble.png")

    # ---- size-only baseline, one per cross-validation run ----
    for k in CV_FOLDS:
        p_test, p_val, val = size_only(labels, k, test)
        threshold = M.youden_threshold(M.p_suspicious(p_val), M.suspicious(val["hard"].to_numpy()))
        m = M.all_metrics(p_test, hard, soft, threshold=threshold)
        m["threshold_from_val"] = m.pop("threshold")
        rows.append({"run": f"size_only_fold{k}", "model": "size only", "fold": k} | m)

    per_run = pd.DataFrame(rows)
    per_run.to_csv(args.out / "per_run.csv", index=False)

    # ---- mean and SD over the 4 folds (ensembles have one row, so no SD) ----
    summary = per_run.groupby("model", sort=False)[METRIC_COLS].agg(["mean", "std"]).round(3)
    summary.to_csv(args.out / "summary.csv")

    # ---- B - A with 95% bootstrap intervals, resampling whole patients ----
    # bootstrap_diff returns metric(first) - metric(second): soft first gives B - A
    soft_runs = np.stack([test_probs["soft", k] for k in CV_FOLDS])
    hard_runs = np.stack([test_probs["hard", k] for k in CV_FOLDS])
    diffs = []
    for name, fn in BOOTSTRAP.items():
        mean, (lo, hi) = M.bootstrap_diff(fn, soft_runs, hard_runs, hard, soft, n=args.resamples,
                                          seed=0, groups=test["patient_id"].to_numpy())
        diffs.append({"metric": name, "b_minus_a": round(mean, 4), "ci_low": lo, "ci_high": hi,
                      "interval_excludes_0": bool(lo > 0 or hi < 0)})
        print(f"  bootstrap {name} done", flush=True)
    b_minus_a = pd.DataFrame(diffs)
    b_minus_a.to_csv(args.out / "b_minus_a.csv", index=False)

    # ---- confidence by reader disagreement ----
    spread = np.minimum(test["spread"].to_numpy(), 3)
    readers = test[SOFT].max(axis=1).groupby(spread).mean()
    cbs_rows = []
    for labels_type, model in MODELS.items():
        per_fold = [M.confidence_by_spread(test_probs[labels_type, k], spread) for k in CV_FOLDS]
        for g, name in zip(SPREAD_GROUPS, SPREAD_NAMES):
            conf = [r[g][0] for r in per_fold]
            cbs_rows.append({"model": model, "spread": name, "nodules": per_fold[0][g][1],
                             "mean_confidence": round(float(np.mean(conf)), 3),
                             "lowest_fold": round(min(conf), 3), "highest_fold": round(max(conf), 3),
                             "readers_agreement": round(float(readers.loc[g]), 3)})
    cbs = pd.DataFrame(cbs_rows)
    cbs.to_csv(args.out / "confidence_by_spread.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for labels_type, model, colour in (("hard", "A (hard labels)", "tab:blue"),
                                       ("soft", "B (soft labels)", "tab:orange")):
        conf = np.array([[M.confidence_by_spread(test_probs[labels_type, k], spread)[g][0]
                          for g in SPREAD_GROUPS] for k in CV_FOLDS])
        ax.plot(SPREAD_GROUPS, conf.mean(axis=0), marker="o", color=colour, label=model)
        ax.fill_between(SPREAD_GROUPS, conf.min(axis=0), conf.max(axis=0), color=colour, alpha=0.15)
    ax.plot(SPREAD_GROUPS, readers.loc[SPREAD_GROUPS].to_numpy(), "k--", marker="s",
            label="readers' own agreement (ideal)")
    counts = cbs[cbs["model"] == MODELS["hard"]]["nodules"].tolist()
    ax.set_xticks(SPREAD_GROUPS, [f"{n}\nn = {c}" for n, c in zip(SPREAD_NAMES, counts)])
    ax.set_xlabel("reader disagreement (highest rating minus lowest)")
    ax.set_ylabel("model's confidence (its top probability)")
    ax.set_title("Model confidence by reader disagreement\n"
                 "TEST fold 0, mean of the 4 cross-validation models (shaded: lowest to highest)", fontsize=10)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(args.out / "confidence_by_spread.png", dpi=150)
    plt.close(fig)

    # ---- print ----
    pd.set_option("display.width", 200)
    print("\nper run (test fold 0):")
    print(per_run.drop(columns=["model"]).round(3).to_string(index=False))
    print("\nmean and SD over the 4 folds:")
    print(summary.to_string())
    print("\nB (soft) - A (hard), 95% bootstrap interval over patients:")
    print(b_minus_a.to_string(index=False))
    print("\nconfidence by reader disagreement:")
    print(cbs.to_string(index=False))
    print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()

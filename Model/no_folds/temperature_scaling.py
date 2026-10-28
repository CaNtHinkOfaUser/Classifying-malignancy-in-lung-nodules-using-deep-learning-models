"""Temperature scaling of the no-folds runs: does it fix Model B's calibration, and at what cost?

Each run gets one temperature T, fitted on its OWN validation predictions and applied
unchanged to its test predictions (same rule as the sensitivity/specificity cut-off in
score_test_no_fold.py). The saved probabilities are turned back into logits with log(p),
divided by T and put through softmax again. T < 1 makes a model more confident, T > 1 less.
argmax never changes, so balanced accuracy and quadratic kappa are identical before and after.

Two ways to fit T, both on validation:
  hard: minimise cross-entropy to the median rating (standard temperature scaling)
  soft: minimise cross-entropy to the readers' rating shares (keeps Model B's own target)

Writes into --out:
    temperatures.csv     T for every run and fitting target
    summary.csv          mean and SD over seeds, per model and version (raw / T-hard / T-soft)
    b_minus_a.csv        B - A with 95% patient-level bootstrap intervals, per version
    confidence_by_spread.csv

Run from the project folder:
    .venv/bin/python Model/no_folds/temperature_scaling.py --runs-dir runs/cls_no_fold_mask_weighted
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "pipeline"))     # metrics.py, ct.py
import metrics as M                    # noqa: E402
from ct import NODULES                 # noqa: E402

PROJECT = Path(__file__).resolve().parents[2]
SEEDS = (1, 2, 3)
MODELS = {"hard": "A (hard)", "soft": "B (soft)"}
PROBS = [f"p{k}" for k in range(1, 6)]
SOFT = [f"soft_{k}" for k in range(1, 6)]
SPREAD_NAMES = ["0 (agreed)", "1", "2", "3+ (disagreed most)"]
BOOTSTRAP = {
    "ce_vs_readers": lambda p, h, s: -(s * np.log(np.clip(p, M.EPS, 1))).sum(axis=1).mean(),
    "ece": lambda p, h, s: M.ece(p, h),
    "auc": lambda p, h, s: M.roc_auc_score(M.suspicious(h), M.p_suspicious(p)),
}


def parse_args():
    p = argparse.ArgumentParser(description="Temperature scaling of the no-folds runs.")
    p.add_argument("--runs-dir", type=Path, default=PROJECT / "runs" / "cls_no_fold_mask_weighted")
    p.add_argument("--out", type=Path, default=PROJECT / "results" / "test_no_fold_mask_weighted_temperature")
    p.add_argument("--resamples", type=int, default=2000)
    return p.parse_args()


def scale(probs, t):
    """softmax(log(p) / T): the saved probabilities re-tempered."""
    z = np.log(np.clip(probs, M.EPS, 1)) / t
    z -= z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def fit_t(probs, target):
    """T in [0.05, 20] minimising cross-entropy to `target` (one-hot or soft, shape N x 5)."""
    loss = lambda log_t: -(target * np.log(np.clip(scale(probs, np.exp(log_t)), M.EPS, 1))).sum(1).mean()
    return float(np.exp(minimize_scalar(loss, bounds=(np.log(0.05), np.log(20)), method="bounded").x))


def load(path, keys):
    return pd.read_csv(path).set_index("nodule_key").loc[keys, PROBS].to_numpy()


def main():
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    labels = pd.read_csv(NODULES)
    test = labels[labels["split"] == "test"].sort_values("nodule_key").reset_index(drop=True)
    val = labels[labels["split"] == "val"].sort_values("nodule_key").reset_index(drop=True)
    hard, soft = test["hard"].to_numpy(), test[SOFT].to_numpy()
    v_hard, v_soft = val["hard"].to_numpy(), val[SOFT].to_numpy()
    v_onehot = np.eye(5)[v_hard - 1]

    rows, temps, probs = [], [], {}
    for lt, model in MODELS.items():
        for seed in SEEDS:
            run = args.runs_dir / f"{lt}_seed{seed}"
            p_val, p_test = load(run / "val_preds.csv", val["nodule_key"]), load(run / "test_preds.csv", test["nodule_key"])
            ts = {"raw": 1.0, "T-hard": fit_t(p_val, v_onehot), "T-soft": fit_t(p_val, v_soft)}
            for version, t in ts.items():
                pv, pt = scale(p_val, t), scale(p_test, t)
                thr = M.youden_threshold(M.p_suspicious(pv), M.suspicious(v_hard))
                m = M.all_metrics(pt, hard, soft, threshold=thr)
                rows.append({"model": model, "version": version, "seed": seed, "T": round(t, 3)}
                            | {k: m[k] for k in ("auc", "balanced_accuracy", "quadratic_kappa",
                                                 "ce_vs_readers", "ece", "sensitivity", "specificity")})
                probs[lt, version, seed] = pt
                if version != "raw":
                    temps.append({"model": model, "seed": seed, "fit_to": version, "T": round(t, 3)})

    per_run = pd.DataFrame(rows)
    per_run.to_csv(args.out / "per_run.csv", index=False)
    pd.DataFrame(temps).to_csv(args.out / "temperatures.csv", index=False)
    cols = ["T", "auc", "balanced_accuracy", "quadratic_kappa", "ce_vs_readers", "ece", "sensitivity", "specificity"]
    summary = per_run.groupby(["model", "version"])[cols].agg(["mean", "std"]).round(3)
    summary.to_csv(args.out / "summary.csv")
    print(summary.to_string(), "\n")

    # B - A, each model in the same version, patient-level bootstrap as in score_test_no_fold.py
    diffs = []
    for version in ("raw", "T-hard", "T-soft"):
        b = np.stack([probs["soft", version, s] for s in SEEDS])
        a = np.stack([probs["hard", version, s] for s in SEEDS])
        for name, fn in BOOTSTRAP.items():
            mean, (lo, hi) = M.bootstrap_diff(fn, b, a, hard, soft, n=args.resamples, seed=0,
                                              groups=test["patient_id"].to_numpy())
            diffs.append({"version": version, "metric": name, "b_minus_a": round(mean, 4), "ci_low": lo,
                          "ci_high": hi, "interval_excludes_0": bool(lo > 0 or hi < 0)})
    b_minus_a = pd.DataFrame(diffs)
    b_minus_a.to_csv(args.out / "b_minus_a.csv", index=False)
    print(b_minus_a.to_string(index=False), "\n")

    # confidence by reader disagreement
    spread = np.minimum(test["spread"].to_numpy(), 3)
    readers = test[SOFT].max(axis=1).groupby(spread).mean()
    cbs = []
    for lt, model in MODELS.items():
        for version in ("raw", "T-hard", "T-soft"):
            per_seed = [M.confidence_by_spread(probs[lt, version, s], spread) for s in SEEDS]
            for g, name in enumerate(SPREAD_NAMES):
                cbs.append({"model": model, "version": version, "spread": name,
                            "mean_confidence": round(float(np.mean([r[g][0] for r in per_seed])), 3),
                            "readers_agreement": round(float(readers.loc[g]), 3)})
    cbs = pd.DataFrame(cbs)
    cbs.to_csv(args.out / "confidence_by_spread.csv", index=False)
    print(cbs.pivot_table(index=["model", "version"], columns="spread", values="mean_confidence").to_string())


if __name__ == "__main__":
    main()

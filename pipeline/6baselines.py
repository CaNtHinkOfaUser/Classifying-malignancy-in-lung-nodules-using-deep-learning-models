# Ishaan

"""Step 6. The numbers a CNN has to beat, computed BEFORE writing one.

Run after:  build_patches.py (needs diameter_mm)
Prints:     for VAL (and TEST only with --test, once, at the very end):
              * always-predict-3            the accuracy floor
              * train-set label frequencies  the "knows nothing" floor for CE / ECE
              * diameter only                multinomial logistic regression on log(diameter_mm)
              * one radiologist vs the rest  a human reference, on 4-reader nodules
Writes:     results/baselines_val_no_fold.csv (or _test_no_fold.csv)

Uses the single split in nodules_labels.csv's `split` column (patient_split.csv),
so these are the numbers the no-folds models in Model/no_folds/ have to beat.
The folds baselines are a separate file, results/baselines_val_per_run.csv.

The diameter model is also your PyTorch warm-up: nn.Linear(1, 5) on the
standardised log-diameter, trained with CrossEntropyLoss, should land within
~0.01 AUC of the sklearn numbers printed here. If it doesn't, the bug is in
your training loop -- and it's far easier to find on 10 parameters than in a CNN.
"""

import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

import metrics as M
from ct import NODULES, PROJECT

EVAL = "test" if "--test" in sys.argv else "val"
SOFT = [f"soft_{k}" for k in range(1, 6)]

df = pd.read_csv(NODULES)
tr, ev = df[df.split == "train"], df[df.split == EVAL]
hard, soft = ev.hard.to_numpy(), ev[SOFT].to_numpy()
y_bin = M.suspicious(hard)

rows = {}

# 1. always 3 (as one-hot it has infinite cross-entropy, so accuracy only)
rows["always_3"] = {"accuracy": (hard == 3).mean(), "balanced_accuracy": 0.2}

# 2. predict the train label frequencies for every nodule
prior = np.bincount(tr.hard, minlength=6)[1:] / len(tr)
p = np.tile(prior, (len(ev), 1))
rows["train_prior"] = M.scores(p, hard, soft) | {"quadratic_kappa": M.quadratic_kappa(p, hard)}

# 3. diameter only
mu, sd = np.log(tr.diameter_mm).mean(), np.log(tr.diameter_mm).std()
x_tr = ((np.log(tr.diameter_mm) - mu) / sd).to_numpy()[:, None]
x_ev = ((np.log(ev.diameter_mm) - mu) / sd).to_numpy()[:, None]
lr = LogisticRegression(C=1e6, max_iter=5000).fit(x_tr, tr.hard)   # C huge = no regularisation
p = lr.predict_proba(x_ev)
score = M.p_suspicious(p)
val_df = df[df.split == "val"]
x_val = ((np.log(val_df.diameter_mm) - mu) / sd).to_numpy()[:, None]
thr = M.youden_threshold(M.p_suspicious(lr.predict_proba(x_val)), M.suspicious(val_df.hard.to_numpy()))
rows["diameter_only"] = (M.scores(p, hard, soft) | {"quadratic_kappa": M.quadratic_kappa(p, hard)}
                         | M.binary(score, y_bin, thr))

# 4. one radiologist vs the median of the other three (4-reader nodules only)
four = ev[ev.num_readers == 4]
own, rest = [], []
for s in four.ratings:
    r = [int(x) for x in s.split(";")]
    for i in range(4):
        others = sorted(r[:i] + r[i + 1:])
        own.append(r[i]); rest.append(others[1])          # median of 3
own, rest = np.array(own), np.array(rest)
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score
rows["one_reader_vs_rest"] = {
    "accuracy": (own == rest).mean(),
    "balanced_accuracy": balanced_accuracy_score(rest, own),
    "quadratic_kappa": cohen_kappa_score(rest, own, weights="quadratic"),
    "mae_rating": np.abs(own - rest).mean(),
    "sensitivity": ((own >= 4) & (rest >= 4)).sum() / max((rest >= 4).sum(), 1),
    "specificity": ((own <= 3) & (rest <= 3)).sum() / max((rest <= 3).sum(), 1),
}

out = pd.DataFrame(rows).T
cols = ["accuracy", "balanced_accuracy", "macro_f1", "quadratic_kappa", "mae_rating",
        "ce_vs_readers", "nll_vs_median", "ece", "auc", "sensitivity", "specificity",
        "precision", "npv", "f1", "binary_accuracy", "threshold"]
out = out[[c for c in cols if c in out.columns]]
(PROJECT / "results").mkdir(exist_ok=True)
out.to_csv(PROJECT / "results" / f"baselines_{EVAL}_no_fold.csv")    # not baselines_val.csv: that holds the folds numbers

pd.set_option("display.width", 200)
print(f"{EVAL}: {len(ev)} nodules, {int(y_bin.sum())} suspicious, "
      f"{len(four)} with 4 readers\n")
print(out.astype(float).round(3).to_string())
print(f"\ndiameter model: log-diameter mean {mu:.4f}, sd {sd:.4f} (standardise with THESE, from train)")
print("sklearn coefficients (per class 1..5):", lr.coef_.ravel().round(3), "intercepts", lr.intercept_.round(3))

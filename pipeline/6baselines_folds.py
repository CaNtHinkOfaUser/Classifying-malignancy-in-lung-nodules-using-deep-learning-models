# Ishaan

"""Step 6b. The size-only baseline for each cross-validation run (folds 1-4), on its VALIDATION fold.

Run after:  General preprocessing/patient_folds.py, from the project folder:
            .venv/bin/python pipeline/6baselines_folds.py
Writes:     results/baselines_val_per_run.csv   one row per run: val_fold, nodules and the
            four headline metrics. Model/plots.py prints these next to each run as "size only".

For run k: a logistic regression on the standardised log-diameter, trained on
that run's training folds (folds 1-4 except k), scored on fold k. Fold 0, the
locked test set, is never read; its size-only numbers come from
Model/folds/score_test_fold.py at the very end.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

import metrics as M
from ct import CV_FOLDS, NODULES, PROJECT, TEST_FOLD

SOFT = [f"soft_{k}" for k in range(1, 6)]

df = pd.read_csv(NODULES)
rows = []
for k in CV_FOLDS:
    train = df[~df["fold"].isin([TEST_FOLD, k])]
    val = df[df["fold"] == k]
    x = np.log(train["diameter_mm"].to_numpy())
    mu, sd = x.mean(), x.std()                       # from the training folds only
    clf = LogisticRegression(max_iter=1000).fit(((x - mu) / sd)[:, None], train["hard"].to_numpy())
    probs = clf.predict_proba(((np.log(val["diameter_mm"].to_numpy()) - mu) / sd)[:, None])
    rows.append({"val_fold": k, "nodules": len(val)} | M.scores(probs, val["hard"].to_numpy(), val[SOFT].to_numpy()))

out = pd.DataFrame(rows)
path = PROJECT / "results" / "baselines_val_per_run.csv"
out.to_csv(path, index=False)
print(out.round(3).to_string(index=False))
print(f"-> {path}")

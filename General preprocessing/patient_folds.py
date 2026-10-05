# Written by Claude for Ishaan

"""Step 3b. Put every patient in one of 5 folds for the classifier's cross-validation.

Run after:  label generation and Patient_split.ipynb, from the project folder:
            .venv/bin/python "General preprocessing/patient_folds.py"
Writes:     data/patient_folds.csv   patient_id -> fold 0-4, the classifier's source of truth
            a `fold` column in data/nodules_labels.csv. The `split` column stays,
            because YOLO still uses that split.
Rerunning gives the same folds (fixed seed). Never edit the file by hand.

How the folds are used is one rule, ct.fold_role: fold 0 is the locked test
set (20% of patients), untouched until one final evaluation at the very end.
Cross-validation runs on folds 1-4: the run for fold k picks its best epoch and
threshold on fold k and trains on the other three.
"""

from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "data"
NODULES = DATA / "nodules_labels.csv"
FOLDS = DATA / "patient_folds.csv"
N_FOLDS = 5
SEED = 42

nodules = pd.read_csv(NODULES)

# Folds are made of whole PATIENTS, never nodules or scans: all of a patient's
# nodules (and both scans, for the 7 patients who have two) share one fold.
patients = (nodules.groupby("patient_id")
                   .agg(stratum=("hard", "max"),        # their most suspicious nodule
                        n_nodules=("hard", "size"))
                   .reset_index())

# Balanced and stratified. Within each stratum, shuffle the patients, take the
# ones with the most nodules first, and give each to the fold that has the
# fewest nodules from this stratum so far (ties: fewest nodules overall, then
# the lowest fold number). Plain round-robin left folds between 345 and 398
# nodules; this gives 377 in every fold and the same label mix.
rng = np.random.default_rng(SEED)
assign = {}
total = np.zeros(N_FOLDS)
for stratum, group in patients.groupby("stratum"):
    group = group.sample(frac=1, random_state=int(rng.integers(1 << 31)))
    group = group.sort_values("n_nodules", ascending=False, kind="stable")
    load = np.zeros(N_FOLDS)
    for pid, count in zip(group["patient_id"], group["n_nodules"]):
        fold = int(np.lexsort((np.arange(N_FOLDS), total, load))[0])
        assign[pid] = fold
        load[fold] += count
        total[fold] += count

folds = pd.DataFrame({"patient_id": list(assign), "fold": list(assign.values())})
folds = folds.sort_values("patient_id")
folds.to_csv(FOLDS, index=False)

nodules["fold"] = nodules["patient_id"].map(assign)
nodules.to_csv(NODULES, index=False)

# ---- checks ----
assert nodules["fold"].notna().all(), "a nodule has no fold"
assert (nodules.groupby("patient_id")["fold"].nunique() == 1).all(), "a patient is in two folds"

summary = nodules.groupby("fold").agg(
    patients=("patient_id", "nunique"),
    nodules=("hard", "size"),
    rated_1=("hard", lambda h: int((h == 1).sum())),
    rated_2=("hard", lambda h: int((h == 2).sum())),
    rated_3=("hard", lambda h: int((h == 3).sum())),
    rated_4=("hard", lambda h: int((h == 4).sum())),
    rated_5=("hard", lambda h: int((h == 5).sum())),
    suspicious=("hard", lambda h: round(float((h >= 4).mean()), 3)),
    spread_2plus=("spread", lambda s: round(float((s >= 2).mean()), 3)),
    spread_4=("spread", lambda s: int((s == 4).sum())),
)
print(summary.to_string())
print(f"\n{len(folds)} patients, {len(nodules)} nodules -> {FOLDS}")
print(f"added 'fold' to {NODULES} (columns now: {len(nodules.columns)})")

#Ishaan
#Patient split -- run ONCE, then never again. Every model reads the saved file.
import numpy as np
import pandas as pd

NODULES   = "data/nodules_labels.csv"
SPLIT_MAP = "data/patient_split.csv"     # patient_id -> split, the single source of truth
SEED      = 42
FRACTIONS = (0.70, 0.15, 0.15)           # train, val, test

nodules = pd.read_csv(NODULES)

# Split by PATIENT, never by nodule or series. A patient can have several nodules
# and (7 of them here) two scans; if any of their data is in train, none of it may
# be in test, or the model is scored on a body it has already seen.
patients = (nodules.groupby("patient_id")["hard"].max()      # their most suspicious nodule
                   .rename("stratum").reset_index())

# Stratify so each split gets the same mix of easy and suspicious patients.
# Without this, test ends up with ~25 class-5 nodules and the number swings with the seed.
rng = np.random.default_rng(SEED)
assign = {}
for stratum, group in patients.groupby("stratum"):
    pids = group["patient_id"].to_numpy()
    rng.shuffle(pids)
    a = round(FRACTIONS[0] * len(pids))
    b = a + round(FRACTIONS[1] * len(pids))
    for pid in pids[:a]:  assign[pid] = "train"
    for pid in pids[a:b]: assign[pid] = "val"
    for pid in pids[b:]:  assign[pid] = "test"

split_map = pd.DataFrame({"patient_id": list(assign), "split": list(assign.values())})
split_map = split_map.sort_values("patient_id")
split_map.to_csv(SPLIT_MAP, index=False)

nodules["split"] = nodules["patient_id"].map(assign)
nodules.to_csv(NODULES, index=False)

# ---- checkpoints ----
assert nodules["split"].notna().all(), "a nodule has no split"
leak = nodules.groupby("patient_id")["split"].nunique()
assert (leak == 1).all(), f"patient in >1 split: {leak[leak > 1].index.tolist()}"

print("patients per split:\n", split_map["split"].value_counts().reindex(["train","val","test"]))
print("\nnodules per split:\n", nodules["split"].value_counts().reindex(["train","val","test"]))
print("\nshare of each hard class per split:")
print(nodules.groupby("split")["hard"].value_counts(normalize=True).unstack()
             .reindex(["train","val","test"]).round(3).to_string())
print("\nshare suspicious (hard >= 4):")
print((nodules.assign(sus=nodules.hard >= 4).groupby("split")["sus"].mean()
              .reindex(["train","val","test"]).round(3)).to_string())
print("\nshare with reader disagreement (spread >= 2):")
print((nodules.assign(d=nodules.spread >= 2).groupby("split")["d"].mean()
              .reindex(["train","val","test"]).round(3)).to_string())
print(f"\nsaved {SPLIT_MAP} and added 'split' to {NODULES}")

# Ishaan and Emmaus

"""Step 5. Build the YOLO detection dataset: 2.5D slice images + one-class box labels.

Run after:  4abuild_slice_index.py, split_cell.py
Run from:   the pipeline folder that holds ct.py
Reads:      data/labels_bb_a.csv   ALL 2,687 nodules, including 1-reader ones (see below)
            data/slice_index.csv, data/series_index.csv, data/patient_split.csv
Writes:     /Volumes/Expansion/processed/yolo_lidc/
                images/{train,val,test}/<patient>_<series6>_<idx>.png   512x512 RGB
                labels/{train,val,test}/<same name>.txt                   "0 xc yc w h", 0-1
                val_monitor.txt   the balanced part of val, checked by YOLO after every epoch
                dataset.yaml
                yolo_index.csv    copy of data/yolo_index.csv
            data/yolo_index.csv        one row per image: split, series, slice, z, boxes, nodules
            data/yolo_extra_split.csv  frozen split for the patients not in patient_split.csv
Usage:      python 5build_yolo_dataset.py --dry-run   choose the images and print the counts only
            python 5build_yolo_dataset.py             build everything
Takes:      far longer and more disk than the 25 Sep build (29,144 images, ~30 min),
            because val and test now hold every slice. Do a --dry-run first to see the count.

Decisions (fixed here so the detector and the classifier can't drift apart):
  * Positives are every slice any reader outlined a >=3 mm nodule on, 1-reader
    nodules included. A detector's job is to find everything a radiologist
    might call a nodule; leaving the 1-reader ones unlabelled would teach it
    that real nodules are background.
  * Split. Classifier patients keep their patient_split.csv split. The 79
    patients whose nodules all have 1 reader go to train, so the val and test
    nodules are exactly the classifier's. The patients with no nodules at all
    are shuffled into train/val/test in the same proportions (seed 42) and the
    result is saved in yolo_extra_split.csv, so it never changes between runs.
  * Train: every positive slice plus as many negatives, per scan. A negative is
    at least 10 mm from every outlined slice (the slice just past the last
    outline can still show the nodule's edge) AND
      - 85% come from the middle 80% of the scan (mostly lung: vessels and
        scars, the negatives worth learning from),
      - 15% come from the top and bottom 10% (neck, shoulders, upper abdomen),
        so the model has seen those regions before it's run on whole scans.
    Scans with no nodules add NEG_PER_EMPTY_SCAN negatives each.
  * Val and test: EVERY slice of EVERY scan, nodule-free scans included,
    because that's what the detector faces on a real patient. Slices next to a
    nodule stay in, unlabelled, so score detections per nodule, not per slice.
    Validating on all of val after every epoch would be slow, so dataset.yaml
    points YOLO at val_monitor.txt: every positive val slice plus negatives
    picked with the train rules. Final numbers come from all of images/val and
    images/test.
  * Image = the slice below, this slice and the slice above as RGB, fixed lung
    window (ct.slice_rgb). Neighbours are the next slices in z after repeated
    z positions are dropped, not file numbers +-1. The evaluation code must
    build its images with the same function.

Known limitations, not fixed here:
  * "Middle 80%" is by slice position, not a lung mask.
  * Nodules under 3 mm have no box but are still in the images.
  * The 3 channels span ~1 mm on thin-slice scans and ~10 mm on 5 mm scans.
  * Train has no negatives within 10 mm of a nodule, so no practice on the
    slices right beside one.
"""

# Emmaus notes
# The pictures are "rainbowy" and "glowing" because of the 2.5D stuff
# A bunch of the slices are omitted in order for balance between positive and negative slices
# Code takes about 57 minutes to run

import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from PIL import Image

from ct import (BOXES, DATA, SPLIT, YOLO_DIR, load_series_index, load_slice_index,
                read_hu, slice_path, slice_rgb)

SEED = 42   # never mixed with the 25 Sep build
EXTRA_SPLIT = DATA / "yolo_extra_split.csv"
SPLITS = ["train", "val", "test"]
FULL_SCAN_SPLITS = ("val", "test")
NEG_PER_POS = 1.0
NEG_PER_EMPTY_SCAN = 10   # train negatives from each scan with no nodules
NEG_GAP_MM = 10.0
EDGE_FRAC = 0.15          # share of negatives from the top and bottom 10% of the scan
MIN_BOX_PX = 3            # a 0-px sliver at the end of a nodule becomes a 3-px box
Z_MATCH_MM = 0.1


def image_name(patient_id, series_uid, idx):
    return f"{patient_id}_{series_uid[-6:]}_{idx:04d}"


def write_image(job):
    """job = (path, rgb array, label lines)."""
    img_path, rgb, lines = job
    Image.fromarray(rgb).save(img_path)
    lbl_path = img_path.parent.parent.parent / "labels" / img_path.parent.name / (img_path.stem + ".txt")
    lbl_path.write_text("".join(lines))      # empty file = background image


def resolve_boxes(boxes, slices_by_series):
    """Put every outline on a real slice by z, never by SOP UID.

    Adds pos (row in the series' slice list, repeated z dropped), idx (file
    number) and z_err. Then merges rows that are the same nodule on the same
    slice: LIDC-IDRI-0017 and -0659 list two SOP UIDs for some slices.
    """
    parts = []
    for uid, g in boxes.groupby("series_instance_uid"):
        zs = slices_by_series[uid]["z"].to_numpy()
        pos = np.abs(zs[None, :] - g["z-slice"].to_numpy()[:, None]).argmin(axis=1)
        parts.append(g.assign(pos=pos, idx=slices_by_series[uid]["idx"].to_numpy()[pos],
                              z_err=np.abs(zs[pos] - g["z-slice"].to_numpy())))
    boxes = pd.concat(parts)
    assert (boxes["z_err"] <= Z_MATCH_MM).all(), "an outline doesn't sit on a real slice"
    return (boxes.groupby(["patient_id", "series_instance_uid", "pos", "idx", "merged_nodule_id"],
                          as_index=False)[["x_centre", "y_centre", "width", "height"]].mean())


def extra_split(series, split_of, boxes, rng):
    """Split for patients missing from patient_split.csv. Reads EXTRA_SPLIT if it exists."""
    if EXTRA_SPLIT.exists():
        return pd.read_csv(EXTRA_SPLIT)
    with_nodules = set(boxes["patient_id"])
    missing = sorted(set(series["patient_id"]) - set(split_of))
    one_reader = [p for p in missing if p in with_nodules]
    no_nodules = [p for p in missing if p not in with_nodules]

    share = pd.Series(split_of).value_counts(normalize=True).reindex(SPLITS, fill_value=0)
    n_val = int(round(share["val"] * len(no_nodules)))
    n_test = int(round(share["test"] * len(no_nodules)))
    shuffled = list(rng.permutation(no_nodules))
    drawn = ["val"] * n_val + ["test"] * n_test + ["train"] * (len(no_nodules) - n_val - n_test)

    return pd.DataFrame({
        "patient_id": one_reader + shuffled,
        "split": ["train"] * len(one_reader) + drawn,
        "reason": ["only 1-reader nodules"] * len(one_reader) + ["no nodules"] * len(no_nodules),
    })


def sample_negatives(zs, pos_z, k, rng):
    """k slice positions at least NEG_GAP_MM from every outlined slice.

    EDGE_FRAC of them come from the top and bottom 10% of the scan, the rest
    from the middle 80%. If one pool runs short the other makes up the number.
    """
    n = len(zs)
    i = np.arange(n)
    if len(pos_z):
        far = np.abs(zs[:, None] - np.asarray(pos_z)[None, :]).min(axis=1) >= NEG_GAP_MM
    else:
        far = np.ones(n, dtype=bool)
    middle = (i >= 0.1 * n) & (i < 0.9 * n)
    mid_pool, edge_pool = i[far & middle], i[far & ~middle]

    k = min(k, len(mid_pool) + len(edge_pool))
    k_edge = min(int(round(EDGE_FRAC * k)), len(edge_pool))
    k_mid = min(k - k_edge, len(mid_pool))
    k_edge = k - k_mid
    return np.concatenate([rng.choice(mid_pool, k_mid, replace=False),
                           rng.choice(edge_pool, k_edge, replace=False)]).astype(int)


def choose_images(boxes, series, slices_by_series, split_of, rng):
    """One row per image to render, for every scan in series_index.csv."""
    box_groups = dict(tuple(boxes.groupby("series_instance_uid")))
    no_boxes = boxes.iloc[0:0]
    rows = []
    for uid, s in series.iterrows():
        pid = s["patient_id"]
        split = split_of[pid]
        sl = slices_by_series[uid]
        zs, idxs = sl["z"].to_numpy(), sl["idx"].to_numpy()
        W, H = s["cols"], s["rows"]

        positives = {}    # pos -> (label lines, nodule ids)
        for p, gs in box_groups.get(uid, no_boxes).groupby("pos"):
            lines = []
            for b in gs.itertuples():
                w = max(b.width + 1, MIN_BOX_PX)       # outline pixels are inclusive
                h = max(b.height + 1, MIN_BOX_PX)
                lines.append(f"0 {b.x_centre / W:.6f} {b.y_centre / H:.6f} {w / W:.6f} {h / H:.6f}\n")
            ids = ";".join(str(int(x)) for x in sorted(gs["merged_nodule_id"].unique()))
            positives[int(p)] = (lines, ids)

        n_neg = int(round(NEG_PER_POS * len(positives))) if positives else NEG_PER_EMPTY_SCAN
        sampled = set(sample_negatives(zs, zs[list(positives)], n_neg, rng).tolist())

        if split in FULL_SCAN_SPLITS:
            keep = range(len(zs))
        else:
            keep = sorted(set(positives) | sampled)
        for p in keep:
            lines, ids = positives.get(p, ([], ""))
            rows.append({
                "patient_id": pid, "series_instance_uid": uid, "split": split,
                "pos": p, "idx": int(idxs[p]), "z": float(zs[p]),
                "n_boxes": len(lines), "nodule_ids": ids, "lines": lines,
                # the balanced subset: every positive and the sampled negatives
                "balanced": p in positives or p in sampled,
            })

    index = pd.DataFrame(rows)
    index["image"] = [image_name(p, u, i) for p, u, i in
                      zip(index.patient_id, index.series_instance_uid, index.idx)]
    return index


def render(index, slices_by_series):
    """Write every PNG and label file, one series at a time so each DICOM is read once."""
    t0 = time.time()
    done = 0
    with ThreadPoolExecutor(8) as pool:
        for k, (uid, g) in enumerate(index.groupby("series_instance_uid", sort=False)):
            pid = g["patient_id"].iloc[0]
            idxs = slices_by_series[uid]["idx"].to_numpy()
            n = len(idxs)
            need = sorted({q for p in g["pos"] for q in (p - 1, p, p + 1) if 0 <= q < n})
            hu = dict(zip(need, pool.map(read_hu, [slice_path(pid, uid, int(idxs[q])) for q in need])))
            jobs = []
            for r in g.itertuples():
                trio = np.stack([hu[max(r.pos - 1, 0)], hu[r.pos], hu[min(r.pos + 1, n - 1)]])
                path = YOLO_DIR / "images" / r.split / f"{r.image}.png"
                jobs.append((path, slice_rgb(trio, 1), r.lines))
            list(pool.map(write_image, jobs))
            done += len(g)
            if k % 50 == 0:
                print(f"  series {k}  images {done}/{len(index)}  {time.time() - t0:.0f}s", flush=True)
    print(f"\nFinished in {time.time() - t0:.0f}s -> {YOLO_DIR}")


if __name__ == "__main__":
    dry_run = "--dry-run" in sys.argv
    if not dry_run and (YOLO_DIR / "images").exists() and any((YOLO_DIR / "images").rglob("*.png")):
        sys.exit(f"{YOLO_DIR} already has images. Rename or delete it first, "
                 "so files from two builds can't mix.")

    split_of = pd.read_csv(SPLIT).set_index("patient_id")["split"].to_dict()
    series = load_series_index()
    slices = load_slice_index().sort_values(["series_instance_uid", "idx"])
    slices_by_series = {uid: g.reset_index(drop=True) for uid, g in slices.groupby("series_instance_uid")}

    boxes = resolve_boxes(pd.read_csv(BOXES), slices_by_series)

    extra = extra_split(series, split_of, boxes, np.random.default_rng(SEED))
    split_of |= extra.set_index("patient_id")["split"].to_dict()
    index = choose_images(boxes, series, slices_by_series, split_of, np.random.default_rng(SEED + 1))

    # ---- what will be built ----
    summary = index.assign(pos_img=index.n_boxes > 0).groupby("split").agg(
        patients=("patient_id", "nunique"), scans=("series_instance_uid", "nunique"),
        images=("image", "size"), with_nodule=("pos_img", "sum"), boxes=("n_boxes", "sum"),
        balanced=("balanced", "sum"))
    print(summary.reindex(SPLITS).to_string())
    print(f"total {len(index)} images (the 25 Sep build had 29,144)")
    print(f"extra patients: {extra.groupby(['reason', 'split']).size().to_dict()}")
    if dry_run:
        sys.exit("dry run: nothing written")

    for split in SPLITS:
        (YOLO_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (YOLO_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)
    if not EXTRA_SPLIT.exists():
        extra.to_csv(EXTRA_SPLIT, index=False)

    render(index, slices_by_series)

    monitor = index[(index.split == "val") & index.balanced]
    (YOLO_DIR / "val_monitor.txt").write_text("".join(f"./images/val/{name}.png\n" for name in monitor.image))
    (YOLO_DIR / "dataset.yaml").write_text(
        f"path: {YOLO_DIR}\n"
        "train: images/train\n"
        "val: val_monitor.txt   # balanced part of val, checked after every epoch\n"
        "test: images/test      # every slice; score all of images/val and images/test at the end\n"
        "names:\n  0: nodule\n")
    out = index.drop(columns=["lines"])
    out.to_csv(DATA / "yolo_index.csv", index=False)
    out.to_csv(YOLO_DIR / "yolo_index.csv", index=False)

    # ---- checkpoints ----
    for split in SPLITS:
        n_img = sum(not p.name.startswith(".") for p in (YOLO_DIR / "images" / split).glob("*.png"))   # skip macOS ._ sidecars
        n_lbl = sum(not p.name.startswith(".") for p in (YOLO_DIR / "labels" / split).glob("*.txt"))
        print(f"{split}: {n_img} png, {n_lbl} txt (MUST both equal {int((index.split == split).sum())})")
    leak = index.groupby("patient_id")["split"].nunique()
    print(f"patients in more than one split: {(leak > 1).sum()} (MUST be 0)")
    full = index[index.split.isin(FULL_SCAN_SPLITS)].groupby("series_instance_uid").size()
    short = sum(full[u] != len(slices_by_series[u]) for u in full.index)
    print(f"val/test scans missing slices: {short} (MUST be 0)")
    n_pos = boxes[["series_instance_uid", "pos"]].drop_duplicates().shape[0]
    print(f"slices with a nodule: {int((index.n_boxes > 0).sum())} (MUST equal {n_pos})")
    print(f"val_monitor.txt: {len(monitor)} images, {int((monitor.n_boxes > 0).sum())} with a nodule")

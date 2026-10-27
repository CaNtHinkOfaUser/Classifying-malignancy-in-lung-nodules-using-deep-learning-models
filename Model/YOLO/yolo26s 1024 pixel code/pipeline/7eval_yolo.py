# Jairus

"""Step 7. Score a trained detector the way it will be used: per nodule, on every
slice of every val or test scan.

Run after:  6train_yolo.py
Run from:   this folder (Model/YOLO/<version>/pipeline); ct.py comes from the project's pipeline/
Reads:      yolo_lidc/images/<split>/*.png and yolo_lidc/yolo_index.csv (from 5build_yolo_dataset.py)
            data/labels_bb_a.csv, data/slice_index.csv, data/series_index.csv
Writes:     <run>/eval_<split>/detections.csv   every box YOLO drew, conf >= CONF_MIN
            <run>/eval_<split>/candidates.csv   detections joined up through the slices
            <run>/eval_<split>/nodules.csv      one row per nodule: found or not, best score
            <run>/eval_<split>/summary.json     the numbers for the report
Usage:      python 7eval_yolo.py --weights <run>/weights/best.pt --split val
                picks the threshold on val (highest one that still finds 90% of nodules)
            python 7eval_yolo.py --weights <run>/weights/best.pt --split test --threshold 0.31
                applies the val threshold to test, once, at the very end
            add --reuse to re-score without running YOLO again (reads detections.csv)

Why not YOLO's own mAP: YOLO scores slices, but a nodule spans several slices
and what matters is whether each nodule is found at all, and how many false
alarms a radiologist would have to look through per scan. So:
  1. YOLO runs on every slice and every box down to CONF_MIN is kept.
  2. Boxes on neighbouring slices at the same spot are joined into one 3D
     candidate (a gap of one slice is allowed). Its score is its best box.
  3. A candidate finds a nodule if one of its boxes has its centre inside that
     nodule's outline box on that slice (+ HIT_PAD_PX). The nodule's score is
     the best of the boxes that land on it, not the candidate's best box.
  4. False positive = a candidate that finds no nodule. FP/scan divides by every
     scan in the split, the nodule-free ones included.
  5. FROC: sensitivity at 1/8 ... 8 FP per scan, and their mean (CPM), the
     LUNA16 measure (Setio et al., 2017), so the numbers compare with papers.
Sensitivity is given for every labelled nodule and for the classifier's
nodules (>= 2 readers), the ones the malignancy CNN will be run on.
Finding nothing where there is nothing matters as much, so specificity,
accuracy and balanced accuracy are also given per slice (every slice of every
scan: does it have a nodule, and did the model draw a box) and per scan (do
the nodule-free scans get no false alarm), plus a table of how sensitivity and
specificity trade off as the threshold changes.
If the model was trained on JPEG images (pack_for_colab.py --jpeg, recorded in
<run>/image_format.txt), every val/test PNG goes through the same JPEG first.

Known limitations:
  * Nodules under 3 mm have no outline, so finding one counts as a false positive.
  * 1-reader nodules count as found when hit and are never false positives.
"""

import argparse
import importlib.util
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from ultralytics import YOLO

from yolo_config import SHARED, YOLO_DIR, to_jpeg      # first: it puts the project's pipeline/ on the path
from ct import BOXES, load_series_index, load_slice_index

KEY = ["series_instance_uid", "merged_nodule_id"]
CONF_MIN = 0.01         # keep everything down to this; the threshold is chosen afterwards
LINK_GAP = 2            # a candidate carries on if its next box is at most 2 slices on
LINK_PAD_PX = 2         # ...and that box's centre is inside the previous box + this
HIT_PAD_PX = 3          # a box finds a nodule if its centre is inside the outline box + this
FROC_RATES = [0.125, 0.25, 0.5, 1, 2, 4, 8]
TARGET_SENS = 0.90      # A1.4: sensitivity above 90%
SIZE_BINS = [0, 6, 10, np.inf]          # mm, for the size breakdown
SIZE_NAMES = ["<6 mm", "6-10 mm", ">=10 mm"]


def load_build_module():
    """5build_yolo_dataset.py starts with a digit, so it can't be imported by name."""
    path = SHARED / "5build_yolo_dataset.py"
    spec = importlib.util.spec_from_file_location("build_yolo_dataset", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def training_format(weights):
    """("png", None) or ("jpeg", quality): what the model was trained on, as
    recorded by 6train_yolo.py. Runs from before that was recorded were PNG."""
    f = Path(weights).parent.parent / "image_format.txt"
    parts = f.read_text().split() if f.exists() else ["png"]
    return (parts[0], int(parts[1])) if parts[0] == "jpeg" else ("png", None)


def load_images(paths, quality):
    """The PNGs as PIL images, put through the same JPEG as the training images."""
    out = []
    for p in paths:
        with Image.open(p) as img:
            out.append(Image.open(io.BytesIO(to_jpeg(img.convert("RGB"), quality))))
    return out


def run_detector(weights, index, root, split, device, batch):
    model = YOLO(weights)
    imgsz = model.ckpt.get("train_args", {}).get("imgsz", 640) if model.ckpt else 640
    fmt, quality = training_format(weights)
    names = index["image"].tolist()
    paths = [str(root / "images" / split / f"{name}.png") for name in names]
    where = index.set_index("image")[["series_instance_uid", "pos"]]
    print(f"running {Path(weights).name} on {len(paths)} {split} slices (imgsz {imgsz}, "
          f"{'as JPEG %d, like the training images' % quality if fmt == 'jpeg' else 'PNG'})")

    # one batch per call: YOLO runs a list of in-memory images as a single batch,
    # so handing it thousands at once would run out of memory
    rows = []
    for start in range(0, len(paths), batch):
        part = paths[start:start + batch]
        source = load_images(part, quality) if fmt == "jpeg" else part
        results = model.predict(source, conf=CONF_MIN, imgsz=imgsz, device=device,
                                batch=batch, verbose=False)
        assert len(results) == len(part), f"YOLO returned {len(results)} results for {len(part)} images"
        for name, r in zip(names[start:start + batch], results):    # results come back in input order
            uid, pos = where.loc[name]
            for (x, y, w, h), c in zip(r.boxes.xywh.cpu().numpy(), r.boxes.conf.cpu().numpy()):
                rows.append((uid, int(pos), float(x), float(y), float(w), float(h), float(c)))
        if (start // batch) % 100 == 0:
            print(f"  {start + len(part)}/{len(paths)}", flush=True)
    return pd.DataFrame(rows, columns=["series_instance_uid", "pos", "x", "y", "w", "h", "conf"])


def link_candidates(dets):
    """Join boxes on neighbouring slices into 3D candidates, one box per slice each."""
    cands = []
    for uid, g in dets.groupby("series_instance_uid"):
        active = []
        for d in g.sort_values(["pos", "conf"], ascending=[True, False]).itertuples(index=False):
            active = [c for c in active if d.pos - c["last"]["pos"] <= LINK_GAP]
            best, best_dist = None, None
            for c in active:
                last = c["last"]
                if last["pos"] == d.pos:
                    continue
                if (abs(d.x - last["x"]) <= last["w"] / 2 + LINK_PAD_PX
                        and abs(d.y - last["y"]) <= last["h"] / 2 + LINK_PAD_PX):
                    dist = (d.x - last["x"]) ** 2 + (d.y - last["y"]) ** 2
                    if best is None or dist < best_dist:
                        best, best_dist = c, dist
            box = {"pos": d.pos, "x": d.x, "y": d.y, "w": d.w, "h": d.h, "conf": d.conf}
            if best is None:
                best = {"series_instance_uid": uid, "boxes": []}
                active.append(best)
                cands.append(best)
            best["boxes"].append(box)
            best["last"] = box
    return cands


def ground_truth(split_series, slices_by_series, build):
    """Outline boxes on real slices (the YOLO labels), plus one row per nodule."""
    raw = pd.read_csv(BOXES)
    raw = raw[raw["series_instance_uid"].isin(split_series)]
    boxes = build.resolve_boxes(raw, slices_by_series)

    series = load_series_index()
    nodules = (raw.assign(side=raw[["width", "height"]].max(axis=1))
                  .groupby(["patient_id"] + KEY)
                  .agg(num_readers=("num_readers", "first"), side_px=("side", "max"))
                  .reset_index())
    nodules["size_mm"] = nodules["side_px"] * series.loc[nodules["series_instance_uid"], "col_spacing"].to_numpy()
    return boxes, nodules.drop(columns="side_px")


def match(cands, boxes):
    """Which nodules each candidate finds, and how confident the boxes ON each
    nodule were. A nodule is credited with the score of the boxes that actually
    land on it, never with a higher-scoring box elsewhere in the same chain."""
    by_slice = {k: g for k, g in boxes.groupby(["series_instance_uid", "pos"])}
    for c in cands:
        hits = {}                                   # nodule id -> best score of a box on it
        for b in c["boxes"]:
            gt = by_slice.get((c["series_instance_uid"], b["pos"]))
            if gt is None:
                continue
            inside = ((np.abs(b["x"] - gt["x_centre"]) <= gt["width"] / 2 + HIT_PAD_PX)
                      & (np.abs(b["y"] - gt["y_centre"]) <= gt["height"] / 2 + HIT_PAD_PX))
            for nid in gt.loc[inside, "merged_nodule_id"].astype(int):
                hits[nid] = max(hits.get(nid, 0.0), b["conf"])
        c["hits"] = hits
    return pd.DataFrame([{
        "series_instance_uid": c["series_instance_uid"],
        "first_pos": c["boxes"][0]["pos"], "last_pos": c["boxes"][-1]["pos"],
        "n_slices": len(c["boxes"]),
        "score": max(b["conf"] for b in c["boxes"]),
        "x": max(c["boxes"], key=lambda b: b["conf"])["x"],
        "y": max(c["boxes"], key=lambda b: b["conf"])["y"],
        "hits": ";".join(str(n) for n in sorted(c["hits"])),
        "hit_scores": ";".join(f"{n}:{c['hits'][n]:.6f}" for n in sorted(c["hits"])),
    } for c in cands], columns=["series_instance_uid", "first_pos", "last_pos", "n_slices",
                                "score", "x", "y", "hits", "hit_scores"])


def best_score_per_nodule(cand, nodules):
    """The best score of a box that landed on each nodule; 0 = never found."""
    found = cand[cand["hits"] != ""].assign(pair=lambda d: d["hit_scores"].str.split(";")).explode("pair")
    found["merged_nodule_id"] = found["pair"].str.split(":").str[0].astype(int)
    found["score"] = found["pair"].str.split(":").str[1].astype(float)
    best = found.groupby(KEY)["score"].max().rename("best_score").reset_index()
    out = nodules.merge(best, on=KEY, how="left")
    out["best_score"] = out["best_score"].fillna(0.0)
    return out


def curve(nod_scores, fp_scores, n_scans):
    """Sensitivity and FP/scan at every threshold, high to low."""
    ts = np.unique(np.concatenate([nod_scores[nod_scores > 0], fp_scores]))[::-1]
    nod_sorted, fp_sorted = np.sort(nod_scores), np.sort(fp_scores)
    sens = (len(nod_sorted) - np.searchsorted(nod_sorted, ts, side="left")) / max(len(nod_sorted), 1)
    fps = (len(fp_sorted) - np.searchsorted(fp_sorted, ts, side="left")) / n_scans
    return ts, sens, fps


def froc(ts, sens, fps):
    at = {r: float(sens[fps <= r].max()) if (fps <= r).any() else 0.0 for r in FROC_RATES}
    return at, float(np.mean(list(at.values())))


def at_threshold(t, nod, fp_scores, n_scans):
    found = nod["best_score"] >= t
    multi = nod["num_readers"] >= 2
    size = pd.cut(nod["size_mm"], SIZE_BINS, labels=SIZE_NAMES, right=False)
    return {
        "threshold": round(float(t), 4),
        "sensitivity_all": round(float(found.mean()), 4),
        "sensitivity_2plus_readers": round(float(found[multi].mean()), 4),
        "fp_per_scan": round(float((fp_scores >= t).sum() / n_scans), 3),
        "nodules_found": f"{int(found.sum())}/{len(nod)}",
        "by_size_2plus_readers": {
            str(s): f"{int(found[multi & (size == s)].sum())}/{int((multi & (size == s)).sum())}"
            for s in SIZE_NAMES},
    }


def slice_and_scan_level(t, index, dets, boxes, cand):
    """Both halves of 'accuracy' at threshold t: finding nodules AND staying quiet
    where there are none. YOLO's box-level confusion matrix can't show the second
    half (it has no 'correctly said nothing' cell), so it is counted per slice
    and per scan here.

    Slice: positive if it has an outline box. A positive slice counts as found
    only if a box >= t lands on the nodule (centre inside the outline + HIT_PAD_PX);
    a negative slice counts as correct only if no box >= t is drawn at all.
    Slices within 2 of an outlined slice are negatives too, but can still show
    the nodule's edge, so specificity is also given without them.
    Scan: a scan with no labelled nodule is correct if no candidate >= t is drawn.
    """
    key = ["series_instance_uid", "pos"]
    slices = index[key].drop_duplicates()
    pos = boxes[key].drop_duplicates().assign(gt=True)
    s = slices.merge(pos, on=key, how="left").fillna({"gt": False})

    d = dets[dets["conf"] >= t]
    s = s.merge(d[key].drop_duplicates().assign(any_box=True), on=key, how="left").fillna({"any_box": False})
    on_nodule = d.merge(boxes, on=key)
    on_nodule = on_nodule[(np.abs(on_nodule["x"] - on_nodule["x_centre"]) <= on_nodule["width"] / 2 + HIT_PAD_PX)
                          & (np.abs(on_nodule["y"] - on_nodule["y_centre"]) <= on_nodule["height"] / 2 + HIT_PAD_PX)]
    s = s.merge(on_nodule[key].drop_duplicates().assign(hit=True), on=key, how="left").fillna({"hit": False})

    near = boxes[key].drop_duplicates()
    near = pd.concat([near.assign(pos=near["pos"] + k) for k in (-2, -1, 1, 2)])
    s = s.merge(near.drop_duplicates().assign(near=True), on=key, how="left").fillna({"near": False})

    gt, hit, box, near = (s[c].astype(bool) for c in ["gt", "hit", "any_box", "near"])
    tp = int((gt & hit).sum()); fn = int((gt & ~hit).sum())
    fp = int((~gt & box).sum()); tn = int((~gt & ~box).sum())
    far = ~gt & ~near
    sens, spec = tp / max(tp + fn, 1), tn / max(tn + fp, 1)
    prec = tp / max(tp + fp, 1)

    scans = index.groupby("series_instance_uid").size().rename("n").to_frame()
    scans["has_nodule"] = scans.index.isin(boxes["series_instance_uid"])
    alarms = cand[cand["score"] >= t].groupby("series_instance_uid").size()
    scans["quiet"] = ~scans.index.isin(alarms.index)
    clean = scans[~scans["has_nodule"]]
    return {
        "slices": {"tp": tp, "fn": fn, "fp": fp, "tn": tn,
                   "sensitivity": round(sens, 4), "specificity": round(spec, 4),
                   "specificity_excluding_slices_next_to_a_nodule": round(float((far & ~box).sum() / max(far.sum(), 1)), 4),
                   "accuracy": round((tp + tn) / len(s), 4), "balanced_accuracy": round((sens + spec) / 2, 4),
                   "precision": round(prec, 4), "f1": round(2 * prec * sens / max(prec + sens, 1e-9), 4)},
        "nodule_free_scans": {"scans": len(clean), "with_no_false_alarm": int(clean["quiet"].sum()),
                              "specificity": round(float(clean["quiet"].mean()), 4) if len(clean) else None},
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--split", choices=["val", "test"], default="val")
    ap.add_argument("--threshold", type=float, default=None, help="the one chosen on val; required for test")
    ap.add_argument("--root", type=Path, default=YOLO_DIR, help="folder made by 5build_yolo_dataset.py")
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--reuse", action="store_true", help="re-score from detections.csv")
    args = ap.parse_args()
    if args.split == "test" and args.threshold is None:
        raise SystemExit("pick the threshold on val first, then pass it with --threshold")

    out = args.weights.parent.parent / f"eval_{args.split}"
    out.mkdir(parents=True, exist_ok=True)

    index = pd.read_csv(args.root / "yolo_index.csv")
    index = index[index["split"] == args.split]
    split_series = set(index["series_instance_uid"])
    n_scans = len(split_series)

    slices = load_slice_index().sort_values(["series_instance_uid", "idx"])
    slices_by_series = {uid: g.reset_index(drop=True) for uid, g in slices.groupby("series_instance_uid")
                        if uid in split_series}
    short = [u for u in split_series if (index["series_instance_uid"] == u).sum() != len(slices_by_series[u])]
    if short:
        print(f"WARNING: {len(short)} {args.split} scans are missing slices; results won't be per whole scan")

    if args.reuse:
        dets = pd.read_csv(out / "detections.csv")
    else:
        dets = run_detector(args.weights, index, args.root, args.split, args.device, args.batch)
        dets.to_csv(out / "detections.csv", index=False)

    boxes, nodules = ground_truth(split_series, slices_by_series, load_build_module())
    cand = match(link_candidates(dets), boxes)
    nod = best_score_per_nodule(cand, nodules)
    fp_scores = cand.loc[cand["hits"] == "", "score"].to_numpy()

    multi = nod["num_readers"] >= 2
    summary = {"weights": str(args.weights), "split": args.split, "scans": n_scans,
               "slices": len(index), "nodules": len(nod), "nodules_2plus_readers": int(multi.sum()),
               "detections": len(dets), "candidates": len(cand)}
    for label, scores in [("all", nod["best_score"].to_numpy()),
                          ("2plus_readers", nod.loc[multi, "best_score"].to_numpy())]:
        at, cpm = froc(*curve(scores, fp_scores, n_scans))
        summary[f"froc_{label}"] = {str(r): round(s, 4) for r, s in at.items()}
        summary[f"cpm_{label}"] = round(cpm, 4)
        summary[f"max_sensitivity_{label}"] = round(float((scores > 0).mean()), 4)

    if args.threshold is None:
        # highest threshold that still finds TARGET_SENS of the classifier's nodules
        ts, sens, _ = curve(nod.loc[multi, "best_score"].to_numpy(), fp_scores, n_scans)
        ok = ts[sens >= TARGET_SENS]
        threshold = float(ok.max()) if len(ok) else CONF_MIN
        if not len(ok):
            print(f"WARNING: {TARGET_SENS:.0%} sensitivity is never reached; reporting at conf {CONF_MIN}")
    else:
        threshold = args.threshold
    summary["at_threshold"] = at_threshold(threshold, nod, fp_scores, n_scans)
    summary["at_threshold"].update(slice_and_scan_level(threshold, index, dets, boxes, cand))
    # the trade-off at a range of thresholds, so the balance is a visible choice
    trade_off = []
    for t in sorted({0.1, 0.2, 0.3, 0.4, 0.5, 0.6, round(threshold, 4)}):
        a, b = at_threshold(t, nod, fp_scores, n_scans), slice_and_scan_level(t, index, dets, boxes, cand)
        trade_off.append({"threshold": t, "nodule_sensitivity_2plus": a["sensitivity_2plus_readers"],
                          "fp_per_scan": a["fp_per_scan"], "slice_sensitivity": b["slices"]["sensitivity"],
                          "slice_specificity": b["slices"]["specificity"], "slice_accuracy": b["slices"]["accuracy"],
                          "clean_scan_specificity": b["nodule_free_scans"]["specificity"]})
    summary["trade_off"] = trade_off

    cand.to_csv(out / "candidates.csv", index=False)
    nod.to_csv(out / "nodules.csv", index=False)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\n{args.split}: {n_scans} scans, {len(index)} slices, {len(nod)} nodules "
          f"({int(multi.sum())} with >= 2 readers), {len(cand)} candidates")
    print(f"{'FP/scan':>10} {'sens (all)':>12} {'sens (>=2 readers)':>20}")
    for r in FROC_RATES:
        print(f"{r:>10} {summary['froc_all'][str(r)]:>12.3f} {summary['froc_2plus_readers'][str(r)]:>20.3f}")
    print(f"{'CPM':>10} {summary['cpm_all']:>12.3f} {summary['cpm_2plus_readers']:>20.3f}")
    a = summary["at_threshold"]
    print(f"\nat threshold {a['threshold']}: sensitivity {a['sensitivity_2plus_readers']:.1%} "
          f"(>= 2 readers), {a['sensitivity_all']:.1%} (all), {a['fp_per_scan']} FP/scan")
    print(f"found by size (>= 2 readers): {a['by_size_2plus_readers']}")
    sl, sc = a["slices"], a["nodule_free_scans"]
    print(f"per slice: sensitivity {sl['sensitivity']:.1%}, specificity {sl['specificity']:.1%} "
          f"({sl['specificity_excluding_slices_next_to_a_nodule']:.1%} without the slices next to a nodule), "
          f"accuracy {sl['accuracy']:.1%}, balanced accuracy {sl['balanced_accuracy']:.1%}")
    print(f"           TP {sl['tp']}  FN {sl['fn']}  FP {sl['fp']}  TN {sl['tn']}")
    if sc["scans"]:
        print(f"nodule-free scans with no false alarm: {sc['with_no_false_alarm']}/{sc['scans']}")
    print(f"\n{'threshold':>9} {'nodule sens':>12} {'FP/scan':>8} {'slice sens':>11} {'slice spec':>11} {'slice acc':>10}")
    for r in trade_off:
        print(f"{r['threshold']:>9} {r['nodule_sensitivity_2plus']:>12.1%} {r['fp_per_scan']:>8} "
              f"{r['slice_sensitivity']:>11.1%} {r['slice_specificity']:>11.1%} {r['slice_accuracy']:>10.1%}")
    if args.split == "val":
        print(f"\nfor test: python 7eval_yolo.py --weights {args.weights} --split test --threshold {a['threshold']}")
    print(f"-> {out}")

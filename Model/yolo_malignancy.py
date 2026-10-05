# Ishaan

"""Experiment: can YOLO rate malignancy on its own, and do soft labels help it?

One network finds each nodule AND rates it 1-5, instead of YOLO finding it and
the 3D CNN rating it. Two versions, trained identically except for the class
target:
    hard   one-hot on the median rating (Ultralytics as installed)
    soft   the readers' rating distribution
Both are scored on validation fold 1 with the classifier's four metrics, so the
numbers line up with the CNN's fold-1 runs. Fold 0, the locked test set, is
dropped before anything else happens.

How a soft label gets through Ultralytics without changing its file format.
Each box's class is written as
    hard class (0-4) + pattern / 1000
where `pattern` is a row of soft_table.npy (one row per distinct reader
distribution). Ultralytics turns classes into whole numbers with .long(), so
box matching, mAP and plots all see the hard class. The soft version patches
one function, TaskAlignedAssigner.get_targets, to swap the one-hot target for
that row. The patches live in this process only; the installed library is
not edited.

YOLO scores classes with a sigmoid each, not a softmax, so a box's five
scores don't sum to 1. A nodule's distribution is each slice's five scores
divided by their sum, averaged over the nodule's slices weighted by
confidence. The box used on a slice is the most confident one overlapping
the radiologists' box (IoU >= 0.3), the same "we know where it is" setting
the CNN is scored in.

Data: the 2.5D PNGs already in yolo_lidc, picked by fold so the patients
match the CNN's fold-1 run: train = every slice with a nodule in folds 2-4,
val = every slice with a nodule in fold 1. Slices without a nodule are left
out to fit this Mac; the background around each nodule still teaches "not a
nodule". 1-reader nodules keep their one rating (one-hot in both versions):
they're boxed in the images, so leaving them out would teach YOLO they're
background. Only fold 1's >=2-reader nodules are scored. New label files go
to /Volumes/Expansion/processed/yolo_malig/; no image is copied.

Usage (from the project folder):
    .venv/bin/python Model/yolo_malignancy.py prepare
    .venv/bin/python Model/yolo_malignancy.py train --labels soft --epochs 20
    .venv/bin/python Model/yolo_malignancy.py evaluate --labels soft
    .venv/bin/python Model/yolo_malignancy.py compare
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT / "pipeline"))
sys.path.insert(0, str(PROJECT / "Model"))
import metrics as M                                                    # noqa: E402
from ct import BOXES, DATA, FOLDS, NODULES, YOLO_DIR, load_slice_index  # noqa: E402

OUT = YOLO_DIR.parent / "yolo_malig"    # label files and image lists; the images stay in yolo_lidc
RUNS = PROJECT / "runs" / "yolo_malig"
WEIGHTS = PROJECT / "synthetic_dataset" / "yolo11n.pt"   # COCO-pretrained, as in the synthetic run
VAL_FOLD, TRAIN_FOLDS = 1, (2, 3, 4)
CODE = 1000          # class = hard + pattern / CODE
SIZE = 512           # every LIDC slice is 512 x 512
MATCH_IOU = 0.3
DEVICE = "mps"
SOFT = [f"soft_{k}" for k in range(1, 6)]
PROBS = [f"p{k}" for k in range(1, 6)]
KEY = ["series_instance_uid", "merged_nodule_id"]


def _build_module():
    """pipeline/5build_yolo_dataset.py, so boxes are placed exactly as in the YOLO dataset."""
    spec = importlib.util.spec_from_file_location("build_yolo", PROJECT / "pipeline" / "5build_yolo_dataset.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def prepare():
    b5 = _build_module()
    slices = load_slice_index().sort_values(["series_instance_uid", "idx"])
    by_series = {uid: g.reset_index(drop=True) for uid, g in slices.groupby("series_instance_uid")}
    boxes = b5.resolve_boxes(pd.read_csv(BOXES), by_series)

    fold = pd.read_csv(FOLDS).set_index("patient_id")["fold"]
    boxes["fold"] = boxes["patient_id"].map(fold)
    boxes = boxes[boxes["fold"].isin([VAL_FOLD, *TRAIN_FOLDS])].copy()   # fold 0 is never used
    boxes["split"] = np.where(boxes["fold"] == VAL_FOLD, "val", "train")

    # >=2-reader nodules: the labels the CNN uses. 1-reader nodules: their one rating.
    rated = pd.read_csv(NODULES)[[*KEY, "nodule_key", "hard", *SOFT]]
    one = (pd.read_csv(BOXES).groupby(KEY, as_index=False)
           .agg(num_readers=("num_readers", "first"), ratings=("ratings", "first")))
    one = one[one["num_readers"] == 1]
    r = one["ratings"].astype(int).to_numpy()
    one = one.assign(nodule_key="", hard=r, **{f"soft_{k}": (r == k).astype(float) for k in range(1, 6)})
    labels = pd.concat([rated, one[rated.columns]], ignore_index=True)
    boxes = boxes.merge(labels, on=KEY, how="left", validate="many_to_one")
    assert boxes["hard"].notna().all(), "a boxed nodule has no rating"

    table, pattern = np.unique(boxes[SOFT].to_numpy().round(4), axis=0, return_inverse=True)
    table = table / table.sum(axis=1, keepdims=True)                  # 0.3333 x 3 -> exactly 1
    assert len(table) < CODE
    boxes["cls"] = boxes["hard"].astype(int) - 1 + pattern.reshape(-1) / CODE

    boxes["image"] = [b5.image_name(p, u, i) for p, u, i in
                      zip(boxes["patient_id"], boxes["series_instance_uid"], boxes["idx"])]
    on_disk = pd.read_csv(DATA / "yolo_index.csv").set_index("image")["split"]
    assert boxes["image"].isin(on_disk.index).all(), "a slice with a nodule has no image"
    boxes["path"] = [str(YOLO_DIR / "images" / on_disk[n] / f"{n}.png") for n in boxes["image"]]

    boxes["w"] = np.maximum(boxes["width"] + 1, b5.MIN_BOX_PX)       # same box size rule as the build
    boxes["h"] = np.maximum(boxes["height"] + 1, b5.MIN_BOX_PX)
    boxes["line"] = [f"{c:.6f} {x / SIZE:.6f} {y / SIZE:.6f} {w / SIZE:.6f} {h / SIZE:.6f}\n"
                     for c, x, y, w, h in zip(boxes["cls"], boxes["x_centre"], boxes["y_centre"], boxes["w"], boxes["h"])]

    for split, g in boxes.groupby("split"):
        lbl_dir = OUT / "labels" / split
        lbl_dir.mkdir(parents=True, exist_ok=True)
        for name, gi in g.groupby("image"):
            (lbl_dir / f"{name}.txt").write_text("".join(gi["line"]))
        (OUT / f"{split}.txt").write_text("".join(f"{p}\n" for p in g.drop_duplicates("image")["path"]))
    (OUT / "data.yaml").write_text(f"path: {OUT}\ntrain: train.txt\nval: val.txt\nnames:\n"
                                   + "".join(f"  {k}: rating {k + 1}\n" for k in range(5)))
    np.save(OUT / "soft_table.npy", table)

    v = boxes[(boxes["split"] == "val") & (boxes["nodule_key"] != "")]
    v = v.assign(x1=v["x_centre"] - v["w"] / 2, y1=v["y_centre"] - v["h"] / 2,
                 x2=v["x_centre"] + v["w"] / 2, y2=v["y_centre"] + v["h"] / 2)
    v[["image", "path", "nodule_key", "x1", "y1", "x2", "y2"]].to_csv(OUT / "val_boxes.csv", index=False)

    for split, g in boxes.groupby("split"):
        print(f"{split}: {g['image'].nunique()} images, {len(g)} boxes, "
              f"{g.groupby(KEY).ngroups} nodules ({(g.drop_duplicates(KEY)['nodule_key'] == '').sum()} with 1 reader), "
              f"patients {g['patient_id'].nunique()}")
    print(f"scored nodules (fold {VAL_FOLD}, >=2 readers): {v['nodule_key'].nunique()}")
    print(f"distinct reader distributions: {len(table)} -> {OUT / 'soft_table.npy'}")


def use_our_labels():
    """Point Ultralytics at the label files in OUT instead of the single-class ones beside the images."""
    import ultralytics.data.dataset as yds
    label_of = {}
    for split in ("train", "val"):
        for line in (OUT / f"{split}.txt").read_text().split():
            label_of[Path(line).stem] = str(OUT / "labels" / split / f"{Path(line).stem}.txt")
    default = yds.img2label_paths

    def img2label_paths(img_paths, *args, **kwargs):
        if args or kwargs:
            return default(img_paths, *args, **kwargs)
        return [label_of[Path(p).stem] for p in img_paths]

    yds.img2label_paths = img2label_paths


def use_whole_classes_for_map():
    """Validation compares classes with ==, so drop the pattern fraction there."""
    from ultralytics.models.yolo.detect import DetectionValidator
    prepare_batch = DetectionValidator._prepare_batch

    def floored(self, si, batch):
        out = prepare_batch(self, si, batch)
        out["cls"] = out["cls"].floor()
        return out

    DetectionValidator._prepare_batch = floored


def use_soft_targets(table):
    """The one change soft labels need: the class target becomes the readers' distribution.

    Ultralytics builds a one-hot target on the matched box's class, then scales
    it by how well the predicted box fits (the "alignment"). Here the one-hot
    is replaced by the nodule's reader distribution, before that same scaling,
    so box matching and the box losses are untouched.
    """
    from ultralytics.utils.tal import TaskAlignedAssigner
    table = torch.as_tensor(table, dtype=torch.float32)
    get_targets = TaskAlignedAssigner.get_targets

    def soft_targets(self, gt_labels, gt_bboxes, target_gt_idx, fg_mask):
        target_labels, target_bboxes, _ = get_targets(self, gt_labels, gt_bboxes, target_gt_idx, fg_mask)
        pattern = ((gt_labels - gt_labels.floor()) * CODE).round().long().flatten()      # (b * max boxes)
        offset = torch.arange(self.bs, device=gt_labels.device)[:, None] * self.n_max_boxes
        dist = table.to(gt_labels.device)[pattern[target_gt_idx + offset]]               # (b, anchors, 5)
        return target_labels, target_bboxes, dist * (fg_mask[:, :, None] > 0)

    TaskAlignedAssigner.get_targets = soft_targets


def train(labels, epochs, batch, fraction, name):
    from ultralytics import YOLO
    use_our_labels()
    use_whole_classes_for_map()
    if labels == "soft":
        use_soft_targets(np.load(OUT / "soft_table.npy"))
    model = YOLO(str(WEIGHTS))
    model.train(data=str(OUT / "data.yaml"), imgsz=SIZE, epochs=epochs, batch=batch, fraction=fraction,
                device=DEVICE, workers=2, seed=0,
                hsv_h=0, hsv_s=0, hsv_v=0,          # channels are slices, not colours
                project=str(RUNS), name=name, exist_ok=True)


def _box_iou(boxes, box):
    """IoU of each row of boxes (n, 4) with one box (4,), all x1 y1 x2 y2."""
    lt = torch.maximum(boxes[:, :2], box[:2])
    rb = torch.minimum(boxes[:, 2:], box[2:])
    inter = (rb - lt).clamp(min=0).prod(1)
    area = (boxes[:, 2:] - boxes[:, :2]).prod(1)
    return inter / (area + (box[2:] - box[:2]).prod() - inter)


def evaluate(name, batch=16):
    """Score fold 1's >=2-reader nodules with the four classifier metrics."""
    from ultralytics import YOLO
    run = RUNS / name
    net = YOLO(str(run / "weights" / "best.pt")).model.float().eval().to(DEVICE)
    head = net.model[-1]
    xyxy_out = bool(getattr(head, "end2end", False) or getattr(head, "xyxy", False))
    gt = pd.read_csv(OUT / "val_boxes.csv")
    groups = list(gt.groupby("image"))

    hits = []   # one row per nodule per slice where YOLO put a box on it
    with torch.no_grad():
        for start in range(0, len(groups), batch):
            chunk = groups[start:start + batch]
            x = torch.stack([torch.from_numpy(np.array(Image.open(g["path"].iloc[0]).convert("RGB"))).permute(2, 0, 1)
                             for _, g in chunk]).float().div(255).to(DEVICE)
            y = net(x)[0].float().cpu()                          # (b, 4 + 5, anchors)
            for (_, g), yi in zip(chunk, y):
                b = yi[:4].T
                pred = b if xyxy_out else torch.cat([b[:, :2] - b[:, 2:] / 2, b[:, :2] + b[:, 2:] / 2], 1)
                cls = yi[4:].T                                   # (anchors, 5): one sigmoid per rating
                conf = cls.max(1).values
                for r in g.itertuples():
                    ok = _box_iou(pred, torch.tensor([r.x1, r.y1, r.x2, r.y2])) >= MATCH_IOU
                    if ok.any():
                        i = torch.where(ok, conf, -1.0).argmax()
                        hits.append((r.nodule_key, float(conf[i]), *cls[i].tolist()))

    h = pd.DataFrame(hits, columns=["nodule_key", "conf", *PROBS])
    dist = h[PROBS].to_numpy()
    h[PROBS] = dist / dist.sum(axis=1, keepdims=True) * h[["conf"]].to_numpy()
    agg = h.groupby("nodule_key")[[*PROBS, "conf"]].sum()
    agg[PROBS] = agg[PROBS].div(agg["conf"], axis=0)

    nl = pd.read_csv(NODULES)
    prior = (nl.loc[nl["fold"].isin(TRAIN_FOLDS), "hard"].value_counts(normalize=True)
             .reindex(range(1, 6), fill_value=0).to_numpy())
    preds = agg[PROBS].reindex(gt["nodule_key"].unique())
    missed = int(preds[PROBS[0]].isna().sum())
    preds.loc[preds[PROBS[0]].isna(), PROBS] = prior                 # never boxed: the train prior
    preds = preds.rename_axis("nodule_key").reset_index()
    preds.to_csv(run / "val_preds.csv", index=False)

    d = preds.merge(nl[["nodule_key", "hard", *SOFT]], on="nodule_key", validate="one_to_one")
    probs, hard, soft = d[PROBS].to_numpy(), d["hard"].to_numpy().astype(int), d[SOFT].to_numpy()
    m = {k: float(v) for k, v in M.scores(probs, hard, soft).items()}
    m |= {"nodules": len(d), "never_boxed": missed,
          "slices_used_median": float(h.groupby("nodule_key").size().median())}
    (run / "val_metrics.json").write_text(json.dumps(m, indent=2))

    import matplotlib
    matplotlib.use("Agg")
    from plots import _confusion_png, _reliability_png
    _confusion_png(hard, probs.argmax(axis=1) + 1, f"YOLO {name}: confusion matrix", run / "malignancy_confusion.png")
    _reliability_png(probs, hard, m["ece"], f"YOLO {name}: reliability", run / "malignancy_reliability.png")
    print(f"YOLO {name}, validation fold {VAL_FOLD}: {len(d)} nodules, {missed} never boxed")
    for k in M_KEYS:
        print(f"  {k:20s}{m[k]:.3f}")


M_KEYS = ["balanced_accuracy", "auc", "ce_vs_readers", "ece"]


def compare():
    """One table: YOLO alone vs the CNN vs size only, all on validation fold 1."""
    rows = {}
    for name in ("hard", "soft"):
        f = RUNS / name / "val_metrics.json"
        if f.exists():
            rows[f"YOLO alone, {name}"] = json.loads(f.read_text())
    for name in ("hard", "soft"):    # the real cross-validation runs from Model/train.py
        f = PROJECT / "runs" / "cls" / f"{name}_fold{VAL_FOLD}" / "val_metrics.json"
        if f.exists():
            rows[f"3D CNN, {name}"] = json.loads(f.read_text())
    size = pd.read_csv(PROJECT / "results" / "baselines_val_per_run.csv").set_index("val_fold").loc[VAL_FOLD]
    rows["size only"] = size.to_dict()
    nl = pd.read_csv(NODULES)
    v = nl[nl["fold"] == VAL_FOLD][SOFT].to_numpy()
    floor = float(-(v * np.log(np.clip(v, 1e-12, 1))).sum(axis=1).mean())
    rows["chance / best possible"] = {"balanced_accuracy": 0.2, "auc": 0.5, "ce_vs_readers": floor, "ece": 0.0}
    table = pd.DataFrame(rows).T[M_KEYS]
    table.to_csv(RUNS / f"compare_fold{VAL_FOLD}.csv")
    print(f"validation fold {VAL_FOLD} (377 nodules, >=2 readers)")
    print(table.round(3).to_string())


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("step", choices=["prepare", "train", "evaluate", "compare"])
    p.add_argument("--labels", choices=["hard", "soft"])
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--fraction", type=float, default=1.0, help="share of train images, for a quick check")
    p.add_argument("--name", help="run folder name (default: the label type)")
    a = p.parse_args()
    if a.step == "prepare":
        prepare()
    elif a.step == "train":
        train(a.labels, a.epochs, a.batch, a.fraction, a.name or a.labels)
    elif a.step == "evaluate":
        evaluate(a.name or a.labels)
    else:
        compare()

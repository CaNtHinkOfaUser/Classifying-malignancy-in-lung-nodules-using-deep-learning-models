# Ishaan

"""End-to-end system: one CT scan in, every nodule YOLO11s finds out, each rated by the 3D CNN.

This joins the two models the project trained and tested separately, using the
same code they were trained with, so a scan goes through exactly the steps the
training data did:

  1. read the DICOM series -> HU volume                   pipeline/ct.py read_hu
  2. every slice -> 2.5D RGB image (z-1, z, z+1)          ct.slice_rgb, lung window
     -> JPEG 95, 4:4:4, like the training images          yolo_config.to_jpeg
  3. YOLO11s on every slice, boxes down to conf 0.01      7eval_yolo.py's settings
  4. boxes on neighbouring slices joined into 3D nodules  7eval_yolo.link_candidates
     kept if the best box scores >= 0.555                the operating threshold chosen on val
  5. 48 mm cube around each nodule's 3D box, 1 mm voxels  ct.extract_cube
     + box mask from YOLO's 2D boxes                      4bbuild_patches.box_mask
  6. centre 40 mm -> NoduleNet3D, 3 seeds averaged        Model/net.py, runs/cls_no_fold_mask_weighted
     Model B (soft labels) is the main answer, Model A (hard labels) is shown beside it

Per nodule it returns the probability of each rating 1-5, the confidence (top
probability), and suspicion: P(4) + P(5) against the mean of the three seeds'
validation thresholds (Youden), the same rule the report's sensitivity and
specificity use.

One difference from training: the cube's mask comes from YOLO's boxes, not the
radiologists' outlines, because a new scan has no outlines. YOLO boxes are about
1 px wider than an outline box (5build_yolo_dataset.py adds 1 px), so the mask is
very slightly larger than the one the CNN was trained with.

Not a medical device. Ratings are trained to agree with LIDC-IDRI radiologists'
opinions, not with biopsy results.

Usage:  .venv/bin/python end_to_end/nodule_pipeline.py <folder or .zip of one CT scan> [--out result.json]
"""

import argparse
import base64
import importlib.util
import io
import json
import sys
import time
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
import torch
from PIL import Image, ImageDraw

PROJECT = Path(__file__).resolve().parents[1]
YOLO_CODE = PROJECT / "Model" / "YOLO" / "yolo11s 1024 pixel code" / "pipeline"
sys.path.insert(0, str(PROJECT / "pipeline"))           # ct.py
sys.path.insert(0, str(PROJECT / "Model"))              # net.py
sys.path.insert(0, str(PROJECT / "Model" / "no_folds"))
sys.path.insert(0, str(YOLO_CODE))                      # yolo_config.py

import ct                                                        # noqa: E402
from dataset_patches_no_fold import CROP, CUBE, scale_hu         # noqa: E402
from metrics import p_suspicious                                 # noqa: E402
from net import NoduleNet3D                                      # noqa: E402
from yolo_config import JPEG_QUALITY, to_jpeg                    # noqa: E402


def _load(name, path):
    """Import a script whose file name starts with a digit (7eval_yolo.py, 4bbuild_patches.py)."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


eval_yolo = _load("eval_yolo", YOLO_CODE / "7eval_yolo.py")
build_patches = _load("build_patches", PROJECT / "pipeline" / "4bbuild_patches.py")

YOLO_WEIGHTS = (PROJECT / "results" / "YOLO results" / "yolo11s 1024 pixel results" / "model"
                / "yolo11s_img1024_e100_b16" / "weights" / "best.pt")
YOLO_SUMMARY = (PROJECT / "results" / "YOLO results" / "yolo11s 1024 pixel results"
                / "results per nodule slice" / "test" / "summary.json")
CNN_RUNS = PROJECT / "runs" / "cls_no_fold_mask_weighted"
SEEDS = (1, 2, 3)
MODELS = {"B": "soft", "A": "hard"}       # B first: soft labels is the system's answer
# Slices YOLO reads at once. At 1024 px each slice needs a lot of GPU memory, and on a
# Mac the GPU shares the RAM: 16 at once ran at 0.3 s/slice with memory free, but at
# ~7 s/slice on an 8 GB Mac that was already swapping. 4 keeps the peak low.
BATCH = 4


def device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# =====================================================================
# 1. Reading a scan
# =====================================================================
def unpack(path, workdir):
    """A .zip is extracted into workdir; a folder is used as it is."""
    path = Path(path)
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as z:
            z.extractall(workdir)
        return Path(workdir)
    return path


def read_scan(folder):
    """The axial CT series with the most slices under `folder`, as an HU volume.

    Real exports mix series (scout images, thick and thin reconstructions, other
    patients' files in a shared folder), so the files are grouped by
    SeriesInstanceUID and the largest axial CT series is used. Slices are sorted
    by z ascending, as the LIDC files were, and a repeated z keeps its first file
    (ct.load_slice_index does the same).
    """
    series = defaultdict(list)
    for f in sorted(p for p in Path(folder).rglob("*") if p.is_file() and not p.name.startswith(".")):
        try:
            ds = pydicom.dcmread(f, stop_before_pixels=True)
        except Exception:
            continue                                  # not DICOM: skip it
        if getattr(ds, "Modality", "") != "CT" or "ImagePositionPatient" not in ds:
            continue
        orient = np.array(getattr(ds, "ImageOrientationPatient", [1, 0, 0, 0, 1, 0]), dtype=float)
        if abs(abs(np.cross(orient[:3], orient[3:])[2]) - 1) > 0.05:
            continue                                  # not an axial slice
        series[ds.SeriesInstanceUID].append((float(ds.ImagePositionPatient[2]), f, ds))
    if not series:
        raise ValueError("no axial CT DICOM slices found in the upload")

    uid, items = max(series.items(), key=lambda kv: len(kv[1]))
    items.sort(key=lambda t: t[0])
    zs, files, first = [], [], items[0][2]
    for z, f, _ in items:
        if not zs or abs(z - zs[-1]) > 1e-3:
            zs.append(z)
            files.append(f)
    if len(files) < 10:
        raise ValueError(f"the largest CT series has only {len(files)} slices: upload a whole scan")

    vol = np.stack([ct.read_hu(f) for f in files])
    row_sp, col_sp = (float(v) for v in first.PixelSpacing)
    zs = np.array(zs)
    return {
        "volume": vol, "zs": zs, "row_spacing": row_sp, "col_spacing": col_sp,
        "z_spacing": float(np.median(np.diff(zs))),
        "series_uid": uid, "n_series_found": len(series),
        "patient_id": str(getattr(first, "PatientID", "")),
        "slice_thickness": float(getattr(first, "SliceThickness", 0) or 0),
        "manufacturer": str(getattr(first, "Manufacturer", "")),
    }


# =====================================================================
# 2-4. Detection
# =====================================================================
def load_yolo():
    from ultralytics import YOLO
    model = YOLO(str(YOLO_WEIGHTS))
    imgsz = model.ckpt.get("train_args", {}).get("imgsz", 1024) if model.ckpt else 1024
    threshold = json.loads(YOLO_SUMMARY.read_text())["at_threshold"]["threshold"]
    return model, imgsz, threshold


def detect(scan, yolo, progress=lambda f, msg, **kw: None, batch=BATCH):
    """Every YOLO box on every slice, then joined into 3D candidates (7eval_yolo.py's rules).

    progress() also gets yolo=(slices done, slices in scan), for the website's slice bar.
    """
    model, imgsz, threshold = yolo
    vol = scan["volume"]
    rows = []
    progress(0.1, f"YOLO11s: slice 0 of {len(vol)}", yolo=(0, len(vol)))
    for start in range(0, len(vol), batch):
        idx = range(start, min(start + batch, len(vol)))
        imgs = [Image.open(io.BytesIO(to_jpeg(Image.fromarray(ct.slice_rgb(vol, i)), JPEG_QUALITY)))
                for i in idx]
        results = model.predict(imgs, conf=eval_yolo.CONF_MIN, imgsz=imgsz, device=device(),
                                batch=batch, verbose=False)
        for i, r in zip(idx, results):
            for (x, y, w, h), c in zip(r.boxes.xywh.cpu().numpy(), r.boxes.conf.cpu().numpy()):
                rows.append(("scan", i, float(x), float(y), float(w), float(h), float(c)))
        progress(0.1 + 0.7 * (idx[-1] + 1) / len(vol), f"YOLO11s: slice {idx[-1] + 1} of {len(vol)}",
                 yolo=(idx[-1] + 1, len(vol)))
    dets = pd.DataFrame(rows, columns=["series_instance_uid", "pos", "x", "y", "w", "h", "conf"])
    cands = eval_yolo.link_candidates(dets) if len(dets) else []
    kept = merge_overlapping([c for c in cands if max(b["conf"] for b in c["boxes"]) >= threshold])
    kept.sort(key=lambda c: -max(b["conf"] for b in c["boxes"]))
    return kept, len(dets), len(cands), threshold


def _extent(c):
    """Slice range and the box around every 2D box of a candidate: (pos0, pos1, x0, y0, x1, y1)."""
    b = c["boxes"]
    return (min(x["pos"] for x in b), max(x["pos"] for x in b),
            min(x["x"] - x["w"] / 2 for x in b), min(x["y"] - x["h"] / 2 for x in b),
            max(x["x"] + x["w"] / 2 for x in b), max(x["y"] + x["h"] / 2 for x in b))


def merge_overlapping(cands):
    """Join kept candidates that are pieces of one nodule.

    link_candidates follows one box per slice, so on a large nodule whose box centre
    drifts between slices the chain can break and restart, giving two or three
    candidates for one nodule. 7eval_yolo.py did not need to merge them (a piece that
    lands on a nodule is never a false positive), but here each would be shown and
    rated as its own nodule. Two candidates are merged when their slice ranges are at
    most LINK_GAP apart and one's centre lies inside the other's box.
    """
    cands = [dict(c, boxes=list(c["boxes"])) for c in cands]
    merged = True
    while merged:
        merged = False
        for i in range(len(cands)):
            for j in range(i + 1, len(cands)):
                a, b = _extent(cands[i]), _extent(cands[j])
                close_z = a[0] - eval_yolo.LINK_GAP <= b[1] and b[0] - eval_yolo.LINK_GAP <= a[1]
                inside = lambda p, q: q[2] <= (p[2] + p[4]) / 2 <= q[4] and q[3] <= (p[3] + p[5]) / 2 <= q[5]
                if close_z and (inside(a, b) or inside(b, a)):
                    cands[i]["boxes"] += cands.pop(j)["boxes"]
                    merged = True
                    break
            if merged:
                break
    for c in cands:     # one box per slice: the most confident
        best = {}
        for x in c["boxes"]:
            if x["pos"] not in best or x["conf"] > best[x["pos"]]["conf"]:
                best[x["pos"]] = x
        c["boxes"] = [best[p] for p in sorted(best)]
    return cands


# =====================================================================
# 5-6. Classification
# =====================================================================
def load_cnns(dev):
    """Three seeds each of Model A and Model B, with each seed's validation threshold."""
    nets = {}
    for name, labels in MODELS.items():
        models, thresholds = [], []
        for s in SEEDS:
            run = CNN_RUNS / f"{labels}_seed{s}"
            net = NoduleNet3D(in_channels=2).to(dev)
            net.load_state_dict(torch.load(run / "best.pt", map_location=dev))
            models.append(net.eval())
            thresholds.append(json.loads((run / "val_metrics.json").read_text())["threshold"])
        nets[name] = (models, float(np.mean(thresholds)))
    return nets


def geometry(cand, scan):
    """The candidate's 3D box, as 4bbuild_patches.py builds it from outline boxes."""
    zs = scan["zs"]
    g = pd.DataFrame([{"z-slice": zs[b["pos"]], "x_min": b["x"] - b["w"] / 2, "x_max": b["x"] + b["w"] / 2,
                       "y_min": b["y"] - b["h"] / 2, "y_max": b["y"] + b["h"] / 2,
                       "pos": b["pos"], "conf": b["conf"], "x": b["x"], "y": b["y"], "w": b["w"], "h": b["h"]}
                      for b in cand["boxes"]])
    centre = ((g["z-slice"].min() + g["z-slice"].max()) / 2,
              (g["y_min"].min() + g["y_max"].max()) / 2,
              (g["x_min"].min() + g["x_max"].max()) / 2)
    return g, centre


def cube_and_mask(scan, g, centre):
    zc, rc, cc = centre
    cube = ct.extract_cube(scan["volume"], scan["zs"], scan["row_spacing"], scan["col_spacing"], zc, rc, cc)
    mask = build_patches.box_mask(g, scan["zs"], scan["row_spacing"], scan["col_spacing"],
                                  scan["z_spacing"], zc, rc, cc)
    return cube, mask


@torch.no_grad()
def classify(cube, mask, nets, dev):
    s = (CUBE - CROP) // 2
    x = np.stack([scale_hu(cube[s:s + CROP, s:s + CROP, s:s + CROP]),
                  mask[s:s + CROP, s:s + CROP, s:s + CROP].astype(np.float32)])
    x = torch.from_numpy(x).unsqueeze(0).to(dev)
    out = {}
    for name, (models, threshold) in nets.items():
        per_seed = np.stack([torch.softmax(m(x), dim=1).cpu().numpy()[0] for m in models])
        p = per_seed.mean(axis=0)
        ps = float(p_suspicious(p[None])[0])
        out[name] = {
            "probabilities": [round(float(v), 4) for v in p],
            "rating": int(p.argmax()) + 1,
            "confidence": round(float(p.max()), 4),
            "p_suspicious": round(ps, 4),
            "suspicion_threshold": round(threshold, 4),
            "suspicious": bool(ps >= threshold),
            "seed_spread": round(float(per_seed.std(axis=0).max()), 4),
        }
    return out


# =====================================================================
# Pictures for the report / website
# =====================================================================
def _png(arr):
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def slice_picture(scan, g):
    """The slice where YOLO was most confident, lung window, with its 2D box, plus a zoom."""
    best = g.loc[g["conf"].idxmax()]
    img = Image.fromarray(ct.to_uint8_window(scan["volume"][int(best["pos"])])).convert("RGB")
    d = ImageDraw.Draw(img)
    box = [best["x_min"] - 2, best["y_min"] - 2, best["x_max"] + 2, best["y_max"] + 2]
    d.rectangle(box, outline=(255, 140, 40), width=2)
    half = max(32, int(max(best["w"], best["h"]) * 1.5))
    cx, cy = int(best["x"]), int(best["y"])
    zoom = img.crop((cx - half, cy - half, cx + half, cy + half)).resize((256, 256), Image.LANCZOS)
    return _png(np.asarray(img)), _png(np.asarray(zoom)), int(best["pos"])


def cube_views(cube, mask):
    """Axial, coronal, sagittal through the cube centre: CT in red, mask in blue (as on the poster)."""
    c = cube.shape[0] // 2
    ctu = (scale_hu(cube) + 1) / 2
    views = []
    for sl in (np.s_[c, :, :], np.s_[::-1, c, :], np.s_[::-1, :, c]):   # z ascending -> flip so head is up
        rgb = np.zeros(ctu[sl].shape + (3,), dtype=np.uint8)
        rgb[..., 0] = (ctu[sl] * 255).astype(np.uint8)
        rgb[..., 2] = (mask[sl] * 200).astype(np.uint8)
        views.append(_png(np.asarray(Image.fromarray(rgb).resize((160, 160), Image.NEAREST))))
    return dict(zip(("axial", "coronal", "sagittal"), views))


# =====================================================================
# The whole thing
# =====================================================================
class Pipeline:
    """Loads both models once; analyse() can then be called for scan after scan."""

    def __init__(self, batch=BATCH):
        self.dev = device()
        self.batch = batch
        self.yolo = load_yolo()
        self.nets = load_cnns(self.dev)

    def analyse(self, folder, progress=lambda f, msg, **kw: None):
        t0 = time.time()
        progress(0.02, "Reading the DICOM files")
        scan = read_scan(folder)
        progress(0.1, f"Read {len(scan['volume'])} slices; running YOLO11s on every slice")
        cands, n_boxes, n_cands, threshold = detect(scan, self.yolo, progress, self.batch)
        if self.dev == "mps":
            torch.mps.empty_cache()          # hand YOLO's working memory back before the CNNs run

        nodules = []
        for k, cand in enumerate(cands, 1):
            progress(0.8 + 0.18 * k / max(len(cands), 1), f"3D CNN: nodule {k} of {len(cands)}")
            g, centre = geometry(cand, scan)
            cube, mask = cube_and_mask(scan, g, centre)
            full, zoom, best_pos = slice_picture(scan, g)
            size_mm = float(max(g["w"].max() * scan["col_spacing"], g["h"].max() * scan["row_spacing"]))
            nodules.append({
                "id": k,
                "yolo_score": round(float(g["conf"].max()), 4),
                "slices": [int(g["pos"].min()) + 1, int(g["pos"].max()) + 1],
                "best_slice": best_pos + 1,
                "z_mm": round(float(centre[0]), 1),
                "centre_px": [round(float(centre[2]), 1), round(float(centre[1]), 1)],
                "size_mm": round(size_mm, 1),
                "extent_z_mm": round(float(g["z-slice"].max() - g["z-slice"].min() + scan["z_spacing"]), 1),
                "boxes": [{"slice": int(r.pos) + 1, "x": round(float(r.x_min), 1), "y": round(float(r.y_min), 1),
                           "w": round(float(r.w), 1), "h": round(float(r.h), 1), "conf": round(float(r.conf), 3)}
                          for r in g.itertuples()],
                "cnn": classify(cube, mask, self.nets, self.dev),
                "images": {"slice": full, "zoom": zoom, **cube_views(cube, mask)},
            })
        nodules.sort(key=lambda n: -n["cnn"]["B"]["p_suspicious"])
        progress(1.0, "Done")
        return {
            "scan": {k: scan[k] for k in ("patient_id", "series_uid", "n_series_found", "row_spacing",
                                          "col_spacing", "z_spacing", "slice_thickness", "manufacturer")}
                    | {"slices": int(len(scan["volume"])), "shape": list(scan["volume"].shape)},
            "detector": {"weights": YOLO_WEIGHTS.name, "threshold": threshold, "boxes": n_boxes,
                         "candidates_all": n_cands, "nodules": len(nodules)},
            "nodules": nodules,
            "seconds": round(time.time() - t0, 1),
            "device": self.dev,
        }


if __name__ == "__main__":
    import tempfile

    p = argparse.ArgumentParser(description="Run YOLO11s + the 3D CNN on one CT scan.")
    p.add_argument("scan", type=Path, help="a folder of DICOM files, or a .zip of one")
    p.add_argument("--out", type=Path, help="write the full result (with images) as JSON")
    p.add_argument("--batch", type=int, default=BATCH, help=f"slices YOLO reads at once (default {BATCH})")
    args = p.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        result = Pipeline(args.batch).analyse(unpack(args.scan, tmp),
                                    progress=lambda f, msg, **kw: print(f"[{f:4.0%}] {msg}", flush=True))
    s, d = result["scan"], result["detector"]
    print(f"\n{s['slices']} slices, {s['row_spacing']:.2f} x {s['col_spacing']:.2f} x {s['z_spacing']:.2f} mm, "
          f"{result['seconds']} s on {result['device']}")
    print(f"YOLO11s: {d['boxes']} boxes -> {d['candidates_all']} 3D candidates -> "
          f"{d['nodules']} at threshold {d['threshold']}")
    for n in result["nodules"]:
        b, a = n["cnn"]["B"], n["cnn"]["A"]
        print(f"  #{n['id']}: slices {n['slices'][0]}-{n['slices'][1]}, ~{n['size_mm']} mm, YOLO {n['yolo_score']:.2f} | "
              f"B rating {b['rating']} (conf {b['confidence']:.2f}), P(susp) {b['p_suspicious']:.2f}"
              f"{' SUSPICIOUS' if b['suspicious'] else ''} | A rating {a['rating']}, P(susp) {a['p_suspicious']:.2f}")
    if args.out:
        args.out.write_text(json.dumps(result, indent=1))
        print("wrote", args.out)

# Jairus

"""YOLO's own evaluation (model.val) of a trained detector on the test set.

This produces exactly what YOLO produces during training: precision, recall,
mAP50, mAP50-95 and its plots (F1 / precision / recall / PR curves, confusion
matrix, example predictions).

Run from:   this folder (Model/YOLO/<version>/pipeline); ct.py comes from the project's pipeline/
Reads:      <root>/images/<split>/*.png and <root>/labels/<split>/*.txt
            (default root: /Volumes/Expansion/processed/yolo_lidc, read only)
Writes:     <run>/<name>/   YOLO's plots, next to the model's run folder
            temporary JPEG copies of the images (deleted at the end)
Usage:      python yolo_val_test.py --weights <run>/weights/best.pt
                conf 0.001 (YOLO's default): use its precision, recall, mAP50, mAP50-95
            python yolo_val_test.py --weights <run>/weights/best.pt --conf 0.29
                a fixed confidence: use ONLY its confusion matrix (mAP needs conf 0.001)
Takes:      ~7 min to prepare the images + ~25 min per run on an M2 MacBook (640;
            about 2.5x longer for a model trained at 1024)

Why JPEG copies: the model was trained on JPEG versions of the images (quality
95, full-resolution colour, see yolo_config.to_jpeg), so the test images are
converted the same way first. Why copies at all: YOLO writes a label cache file
next to the labels, and the hard disk must not be written to.
"""

import argparse
import shutil
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image
from ultralytics import YOLO

from yolo_config import YOLO_DIR, to_jpeg


def prepare(root, split, tmp):
    img_out, lbl_out = tmp / "images" / split, tmp / "labels" / split
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)
    names = sorted(p.stem for p in (root / "images" / split).iterdir()
                   if p.suffix == ".png" and not p.name.startswith("."))

    def convert(n):
        with Image.open(root / "images" / split / f"{n}.png") as im:
            (img_out / f"{n}.jpg").write_bytes(to_jpeg(im.convert("RGB")))
        shutil.copyfile(root / "labels" / split / f"{n}.txt", lbl_out / f"{n}.txt")

    t0 = time.time()
    with ThreadPoolExecutor(8) as pool:
        for k, _ in enumerate(pool.map(convert, names)):
            if k % 8000 == 0:
                print(f"  prepared {k}/{len(names)}  {time.time() - t0:.0f}s", flush=True)
    yaml = tmp / "data.yaml"
    yaml.write_text(f"path: {tmp}\ntrain: images/{split}\nval: images/{split}\n"
                    f"test: images/{split}\nnames:\n  0: nodule\n")
    print(f"prepared {len(names)} {split} images in {(time.time() - t0) / 60:.1f} min")
    return yaml


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--conf", type=float, nargs="*", default=[0.001],
                    help="one or more confidence thresholds; 0.001 = YOLO's default (for mAP)")
    ap.add_argument("--split", default="test")
    ap.add_argument("--root", type=Path, default=YOLO_DIR, help="folder made by 5build_yolo_dataset.py")
    ap.add_argument("--device", default="mps", help="mps on a Mac, 0 on a CUDA GPU, or cpu")
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()

    run = args.weights.parent.parent
    tmp = Path(tempfile.mkdtemp(prefix="yolo_val_"))
    try:
        data = prepare(args.root, args.split, tmp)
        model = YOLO(args.weights)
        # evaluate at the image size the model was trained at (640 for v1, 1024 for an img1024 run)
        imgsz = (model.ckpt or {}).get("train_args", {}).get("imgsz", 640)
        print(f"image size {imgsz} (from the model)")
        for conf in args.conf:
            name = f"val_{args.split}" + ("" if conf == 0.001 else f"_conf{conf}")
            m = model.val(data=str(data), split=args.split, imgsz=imgsz, batch=args.batch,
                          device=args.device, conf=conf, plots=True,
                          project=str(run), name=name, exist_ok=True)
            P, R = m.box.mp, m.box.mr
            print(f"\nconf {conf} -> {run / name}")
            if conf == 0.001:
                print(f"precision {P:.4f}  recall {R:.4f}  F1 {2 * P * R / (P + R):.4f}  "
                      f"mAP50 {m.box.map50:.4f}  mAP50-95 {m.box.map:.4f}")
            cm = getattr(m, "confusion_matrix", None)
            if cm is not None:
                M = cm.matrix
                print(f"confusion matrix: found {int(M[0, 0])}  missed {int(M[1, 0])}  false boxes {int(M[0, 1])}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        print("temporary JPEG copies deleted")

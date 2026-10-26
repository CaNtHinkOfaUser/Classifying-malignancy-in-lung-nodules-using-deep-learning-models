# Jairus

"""Pack what YOLO training needs into two zips for Google Colab.

Run after:  5build_yolo_dataset.py
Run from:   this folder (Model/YOLO/<version>/pipeline); ct.py comes from the project's pipeline/
Writes:     /Volumes/Expansion/processed/colab/yolo_lidc_train_jpg.zip   (~3.9 GB, with --jpeg)
            or                          .../yolo_lidc_train.zip       (~10.5 GB, PNG as built)
                yolo_lidc/images+labels/train   all 22,050 training slices
                yolo_lidc/images+labels/val     only the 4,324 slices in val_monitor.txt
                yolo_lidc/val_monitor.txt, yolo_lidc/yolo_index.csv
                yolo_lidc/image_format.txt      "png" or "jpeg 95"; 6train_yolo.py copies it
                                                into the run so 7eval_yolo.py scores val/test
                                                in the same format
            /Volumes/Expansion/processed/colab/lidc_code.zip         (small)
                code/pipeline/*.py and the data/*.csv the scripts read
Usage:      python pack_for_colab.py --jpeg     both zips, images as JPEG (2.7x smaller upload)
            python pack_for_colab.py            both zips, images as PNG
            python pack_for_colab.py --code     only the code zip (after editing a script)
            add --out <folder> to write the zips somewhere other than the drive
Then:       upload both zips to a "LIDC" folder in Google Drive and open Model/yolo_colab.ipynb in Colab.
            Delete the colab folder from the drive once the upload has finished.

The full val and test sets (every slice, ~64k images) are left out: training
only checks val_monitor.txt, and 7eval_yolo.py can be run on the Mac afterwards.
Images are stored, not compressed again: PNG and JPEG don't shrink further.
JPEG goes through yolo_config.to_jpeg (quality 95, colour at full resolution).
macOS "._" sidecar files on the exFAT drive are skipped.
"""

import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

from yolo_config import JPEG_QUALITY, PROCESSED, SHARED, YOLO_DIR, to_jpeg   # first: it finds ct
from ct import DATA

OUT = PROCESSED / "colab"
SHARED_FILES = ["ct.py", "5build_yolo_dataset.py"]                  # from the project's pipeline/
YOLO_FILES = ["yolo_config.py", "6train_yolo.py", "7eval_yolo.py"]   # from this folder
DATA_FILES = ["labels_bb_a.csv", "slice_index.csv", "series_index.csv", "nodules_labels.csv",
              "patient_split.csv", "yolo_extra_split.csv", "yolo_index.csv"]


def pack_code():
    path = OUT / "lidc_code.zip"
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        for name in SHARED_FILES:
            z.write(SHARED / name, f"code/pipeline/{name}")
        for name in YOLO_FILES:
            z.write(Path(__file__).with_name(name), f"code/pipeline/{name}")
        for name in DATA_FILES:
            z.write(DATA / name, f"code/data/{name}")
    print(f"-> {path} ({path.stat().st_size / 1e6:.0f} MB)")


def read_image(rel, jpeg):
    """(name inside the zip, bytes) for one image."""
    src = YOLO_DIR / rel
    if not jpeg:
        return rel, src.read_bytes()
    with Image.open(src) as img:
        return rel.removesuffix(".png") + ".jpg", to_jpeg(img.convert("RGB"))


def pack_data(jpeg):
    monitor = [line.strip().removeprefix("./") for line in (YOLO_DIR / "val_monitor.txt").read_text().splitlines()
               if line.strip()]
    train = sorted(p.name for p in (YOLO_DIR / "images" / "train").iterdir()
                   if p.suffix == ".png" and not p.name.startswith("."))
    images = [f"images/train/{n}" for n in train] + monitor
    labels = [f.replace("images/", "labels/", 1).removesuffix(".png") + ".txt" for f in images]
    missing = [f for f in images + labels if not (YOLO_DIR / f).exists()]
    if missing:
        sys.exit(f"{len(missing)} files missing, e.g. {missing[:3]}")

    ext, fmt = (".jpg", f"jpeg {JPEG_QUALITY}") if jpeg else (".png", "png")
    path = OUT / ("yolo_lidc_train_jpg.zip" if jpeg else "yolo_lidc_train.zip")
    print(f"packing {len(train)} train + {len(monitor)} val images ({fmt}) and their labels -> {path}")

    t0 = time.time()
    with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED, allowZip64=True) as z, ThreadPoolExecutor(8) as pool:
        for k, (name, data) in enumerate(pool.map(lambda f: read_image(f, jpeg), images)):
            z.writestr(f"yolo_lidc/{name}", data)
            if k % 2000 == 0:
                print(f"  images {k}/{len(images)}  {time.time() - t0:.0f}s", flush=True)
        for f in labels:
            z.write(YOLO_DIR / f, f"yolo_lidc/{f}")
        z.writestr("yolo_lidc/val_monitor.txt", "".join(f"./{f.removesuffix('.png')}{ext}\n" for f in monitor))
        z.write(YOLO_DIR / "yolo_index.csv", "yolo_lidc/yolo_index.csv")
        z.writestr("yolo_lidc/image_format.txt", fmt + "\n")
    print(f"-> {path} ({path.stat().st_size / 1e9:.1f} GB) in {(time.time() - t0) / 60:.0f} min")


if __name__ == "__main__":
    if "--out" in sys.argv:        # e.g. --code --out ../colab_upload, to keep the drive untouched
        OUT = Path(sys.argv[sys.argv.index("--out") + 1]).expanduser()
    OUT.mkdir(parents=True, exist_ok=True)
    pack_code()
    if "--code" not in sys.argv:
        pack_data(jpeg="--jpeg" in sys.argv)

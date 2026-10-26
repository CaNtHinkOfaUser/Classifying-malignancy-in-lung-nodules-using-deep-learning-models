# Jairus

"""Settings for the YOLO training and evaluation scripts (6train_yolo.py,
7eval_yolo.py, pack_for_colab.py) that ct.py doesn't have.

ct.py, 5build_yolo_dataset.py and the data/*.csv files are not copied into this
folder: importing this file puts the project's own pipeline/ folder on the path,
so the YOLO scripts use the same ct.py as everything else. On Colab,
pack_for_colab.py puts them all in one folder, which is found the same way.

ct.py stays the one place for the data paths and for how CT is read; this file
only adds what running on Google Colab needs:
  * the paths can be moved with environment variables, e.g.
        LIDC_PROCESSED=/content/processed LIDC_YOLO_RUNS=/content/drive/MyDrive/LIDC/yolo_runs
    Without them, everything is exactly where ct.py says.
  * where training runs are saved
  * the JPEG settings used to shrink the training images for the upload
"""

import io
import os
import sys
from pathlib import Path

# the nearest folder above this one with pipeline/ct.py: the project root on the Mac, code/ on Colab
for _folder in Path(__file__).resolve().parents:
    if (_folder / "pipeline" / "ct.py").exists():
        sys.path.insert(0, str(_folder / "pipeline"))
        break
else:
    sys.exit("ct.py not found: keep this folder inside the project (Model/YOLO/<version>/pipeline)")

import ct

SHARED = Path(ct.__file__).parent       # the project's pipeline/: ct.py, 5build_yolo_dataset.py

PROCESSED = Path(os.environ.get("LIDC_PROCESSED", ct.PROCESSED))
YOLO_DIR = PROCESSED / ct.YOLO_DIR.name
# one folder per training run; on Colab point this at Google Drive so runs survive a disconnect
YOLO_RUNS = Path(os.environ.get("LIDC_YOLO_RUNS", PROCESSED / "yolo_runs"))

JPEG_QUALITY = 95               # Colab upload: ~37% of the PNG size, mean change ~1.6/255 per pixel


def to_jpeg(img, quality=JPEG_QUALITY):
    """A YOLO image (PIL, RGB) as JPEG bytes. The training images for Colab and
    the val/test images scored afterwards both go through here, so the model is
    tested on exactly the kind of image it was trained on.

    subsampling=0 (4:4:4): R, G and B are three neighbouring slices, so the
    colour must keep full resolution. JPEG's default (4:2:0) halves it, which
    would blur the slice-to-slice differences that tell a nodule from a vessel.
    """
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=quality, subsampling=0)
    return buf.getvalue()

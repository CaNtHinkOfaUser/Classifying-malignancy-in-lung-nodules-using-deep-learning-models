# YOLO results v1: YOLO26s nodule detector

A snapshot of the first trained nodule detector (YOLO26s), with the code, data
tables, model and evaluation results needed to use it or reproduce it.
Everything here is a **copy**; the originals in the project folder,
`~/Downloads/yolo_runs/` and the Expansion hard disk were not changed.

## What's inside

| Folder | Contents |
|---|---|
| `model/yolo26s_img640_e100_b16-2/` | The trained model (`weights/best.pt` = epoch 35, `last.pt` = epoch 55) and everything YOLO saved during training: `results.csv` / `results.png` (loss and metrics per epoch), `args.yaml` (all settings), curves, confusion matrix, example batches. `image_format.txt` records that it was trained on JPEG 95 images. |
| `results/metrics_summary.csv` | **Every number in one table**, with where each comes from. |
| `results/test_yolo_val/` | YOLO's own evaluation on the test set: use for **precision, recall, F1, mAP50, mAP50-95** and the curves. It was run keeping every box down to confidence 0.001; YOLO then reports precision, recall and F1 at the confidence where F1 is highest (0.51), and mAP as the area under the precision-recall curve over all confidences. |
| `results/test_yolo_val_conf0.29/` | The same at confidence 0.29 (chosen on validation): use **only its confusion matrix**. This is the one to report. |
| `results/test_yolo_val_conf0.51/` | The same at confidence 0.51: comparison only (0.51 was picked by looking at the test set). |
| `results/per_nodule_and_specificity/` | Per-nodule sensitivity and per-slice specificity/accuracy (from `7eval_yolo.py`), for val and test. YOLO's own evaluation cannot give specificity. |
| `code/pipeline/` | The scripts (see below). |
| `code/data/` | The CSV tables the scripts read (radiologist annotations, slice index, splits, YOLO image index). |
| `code/notebooks/yolo_colab.ipynb` | The Colab training notebook. |
| `colab_upload/` | `lidc_code.zip` and `dataset.yaml`, as uploaded to `My Drive/LIDC/` for Colab. |

## Main results (test set: 141 scans, 31,655 slices, 2,302 nodule boxes)

From YOLO's own evaluation (`model.val`):

| Metric | Test | During training (val-check set, epoch 35) |
|---|---|---|
| Precision | 0.656 | 0.796 |
| Recall (sensitivity) | 0.555 | 0.719 |
| F1 | 0.60 | 0.76 |
| mAP50 | 0.582 | 0.779 |
| mAP50-95 | 0.357 | 0.473 |

Confusion matrix on test at confidence 0.29: 1,548 nodule boxes found, 754 missed,
1,885 false boxes (about 1 per 17 slices). True negatives are always blank in
object detection.

The test set contains every slice of every scan (about 94% without a nodule),
while the val-check set used during training is balanced, so precision and mAP are
lower on test. See `results/metrics_summary.csv` for all numbers.

## Training settings
YOLO26s, image size 640, batch 16, up to 100 epochs with early stopping
(patience 20; stopped at 55, best epoch 35), seed 42, single class "nodule",
no colour augmentation, zoom 0.25, left-right flips only. Trained on Google
Colab (T4 GPU) on 22,050 JPEG training slices. Full list: `model/.../args.yaml`.

## The code (`code/pipeline/`)
| Script | What it does |
|---|---|
| `ct.py`, `metrics.py`, `4abuild_slice_index.py`, `5build_yolo_dataset.py` | The team's original pipeline files (copies, unchanged) |
| `yolo_config.py` | Data paths (can be moved for Colab) and the JPEG conversion |
| `6train_yolo.py` | Training (used by the Colab notebook) |
| `pack_for_colab.py` | Makes the zips for Google Drive / Colab |
| `yolo_val_test.py` | **YOLO's own evaluation on the test set** (produced the `test_yolo_val*` folders) |
| `7eval_yolo.py` | Per-nodule sensitivity and per-slice specificity |

Run scripts from inside `code/pipeline/`; they find the CSVs in `code/data/` automatically.
They need Python with `ultralytics` (`pip install "ultralytics>=8.4"`).

Example, YOLO's test evaluation (hard disk plugged in):
```
python yolo_val_test.py --weights "../../model/yolo26s_img640_e100_b16-2/weights/best.pt"
python yolo_val_test.py --weights "../../model/yolo26s_img640_e100_b16-2/weights/best.pt" --conf 0.29
```

## The images (not copied: too big for the laptop)
| What | Where |
|---|---|
| All YOLO images and labels (train / val / test, PNG) | Expansion hard disk: `/Volumes/Expansion/processed/yolo_lidc/` |
| Training images as JPEG + labels (what the model trained on), 3.9 GB | Google Drive: `My Drive/LIDC/yolo_lidc_train_jpg.zip` (also on the hard disk in `processed/colab/`, if not deleted) |
| Training runs as saved by Colab | Google Drive: `My Drive/LIDC/yolo_runs/` |

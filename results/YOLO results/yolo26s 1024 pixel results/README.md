# YOLO results v2: YOLO26s at image size 1024

The second nodule detector: the same model (YOLO26s), data, code and settings as
v1, with **one change: the input image size, 1024 instead of 640** (the 512 x 512
slices are enlarged 2x instead of 1.25x). Everything here is a **copy**; the
originals in `~/Downloads/yolo_runs 2/`, the project folder and the Expansion hard
disk were not changed. v1 is in `~/Downloads/yolo results v1/`.

## What's inside

| Folder | Contents |
|---|---|
| `model/yolo26s_img1024_e100_b16/` | The trained model (`weights/best.pt` = epoch 27, `last.pt` = epoch 47) and everything YOLO saved during training (`results.csv`, `results.png`, `args.yaml`, curves, confusion matrix, example batches). |
| `results/metrics_summary.xlsx` (and `.csv`) | **v1 vs v2, every number side by side**, with notes on how each was measured. |
| `results/test_yolo_val/` | YOLO's own evaluation on the test set: precision, recall, F1, mAP50, mAP50-95 and the curves. |
| `results/test_yolo_val_conf0.357/` | The same at confidence 0.357 (v2's F1 peak on validation): use **only its confusion matrix**. |
| `code/`, `colab_upload/` | The code and data tables (same as v1, plus `yolo_val_test.py` and the Kaggle notebook). |

## Training
- Epochs 1-17 on Google Colab, 18-47 on Kaggle (both NVIDIA T4 GPUs), continued from
  the saved checkpoint with every setting unchanged.
- Early stopping (patience 20) ended training at epoch 47; the best epoch was 27.
- Only differences from v1 in `args.yaml`: `imgsz` (640 -> 1024) and `workers` (2 -> 4,
  the number of data-loading processes, which does not affect results).

## Main results (test set: 141 scans, 31,655 slices, 2,302 nodule boxes)

From YOLO's own evaluation (`model.val`):

| Metric | v1 (640) | v2 (1024) |
|---|---|---|
| Precision | 0.656 | 0.654 |
| Recall (sensitivity) | 0.555 | 0.558 |
| F1 | 0.601 | 0.602 |
| mAP50 | 0.582 | 0.584 |
| mAP50-95 | 0.357 | 0.358 |

Confusion matrix at each model's validation F1-peak confidence:

| | v1 (conf 0.29) | v2 (conf 0.357) |
|---|---|---|
| Nodule boxes found | 1,548 | 1,501 |
| Nodule boxes missed | 754 | 801 |
| False boxes | 1,885 | 1,458 |

On the validation-check set used during training, v2 scored higher than v1
(mAP50 0.798 vs 0.779). On the test set, the two models performed almost
identically. That set was also used to choose the best epoch, so its scores are
optimistic for both models; the test set is the fair comparison.

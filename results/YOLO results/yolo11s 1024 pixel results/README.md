# YOLO results v3: YOLO11s at image size 1024

The comparison model from report section B3.2: **YOLO11s**, trained on the same
data with the same code and settings as v2 (YOLO26s at image size 1024), so the
**model is the only difference** from v2. Everything here is a **copy**; the
originals in `~/Downloads/yolo_runs 5/`, the project folder and the Expansion hard
disk were not changed. v1 and v2 are in `yolo results v1` and `yolo results v2`.

## What's inside

| Folder | Contents |
|---|---|
| `model/yolo11s_img1024_e100_b16/` | The trained model (`weights/best.pt` = epoch 36, `last.pt` = epoch 56) and everything YOLO saved during training (`results.csv`, `results.png`, `args.yaml`, curves, confusion matrix, example batches). |
| `results/metrics_summary.xlsx` (and `.csv`) | **v1, v2 and v3 side by side**: every number, with how it was measured. |
| `results/test_yolo_val/` | YOLO's own evaluation on the test set: precision, recall, F1, mAP50, mAP50-95 and the curves. |
| `results/test_yolo_val_conf0.36/` | The same at confidence 0.36 (v3's F1 peak on validation): use **only its confusion matrix**. |
| `code/`, `colab_upload/` | The code and data tables (same as v2). |

## Training
- Trained on Kaggle (NVIDIA T4 GPU) in two sessions: paused after epoch 53 at the
  12-hour session limit, then continued from the checkpoint.
- Early stopping (patience 20) ended training at epoch 56; the best epoch was 36.

## Test set results (141 scans, 31,655 slices, 2,302 nodule boxes), YOLO `model.val`

| Metric | v1 YOLO26s 640 | v2 YOLO26s 1024 | **v3 YOLO11s 1024** |
|---|---|---|---|
| Precision | 0.656 | 0.654 | 0.656 |
| Recall (sensitivity) | 0.555 | 0.558 | **0.567** |
| F1 | 0.601 | 0.602 | **0.608** |
| mAP50 | 0.582 | 0.584 | **0.599** |
| mAP50-95 | 0.357 | 0.358 | **0.366** |

Confusion matrix at each model's validation F1-peak confidence:

| | v1 (0.29) | v2 (0.357) | v3 (0.36) |
|---|---|---|---|
| Nodule boxes found | 1,548 | 1,501 | 1,528 |
| Nodule boxes missed | 754 | 801 | 774 |
| False boxes | 1,885 | 1,458 | 1,431 |

YOLO11s scored slightly higher than both YOLO26s models on every test metric,
although the differences are small (mAP50 +0.015 over v2). On the validation-check
set during training the order was reversed (YOLO26s 0.798 vs YOLO11s 0.789 mAP50);
that set was also used to choose the best epoch, so the test set is the fair
comparison. The v3 test evaluation used Ultralytics 8.4.183 (v1 and v2: 8.4.175),
a minor update of the same release.

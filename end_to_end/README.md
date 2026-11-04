# End-to-end system: CT scan in, rated nodules out

This joins the two models the project trained and tested separately:

```
CT scan (DICOM) ─► every slice as a 2.5D image ─► YOLO11s 2D boxes ─► boxes joined into 3D nodules
                ─► 40 mm cube + box mask per nodule ─► 3D CNN (3 seeds) ─► rating 1–5 probabilities,
                                                                           confidence, suspicion
```

Every step reuses the code the models were trained with, so a new scan is
prepared exactly like the training data:

| Step | Reused from |
|---|---|
| DICOM → HU, lung window, 2.5D RGB slice | `pipeline/ct.py` (`read_hu`, `slice_rgb`) |
| JPEG 95, 4:4:4 (YOLO was trained on JPEGs) | `Model/YOLO/yolo11s 1024 pixel code/pipeline/yolo_config.py` |
| YOLO11s, boxes down to 0.01, joined across slices | `7eval_yolo.py` (`link_candidates`, `CONF_MIN`) |
| Operating threshold 0.555 | `results/YOLO results/yolo11s 1024 pixel results/results per nodule slice/test/summary.json` |
| 48 mm cube, 1 mm voxels, centred on the 3D box | `ct.extract_cube` |
| Box mask on the cube's grid | `pipeline/4bbuild_patches.py` (`box_mask`) |
| Centre 40 mm crop, HU → −1..1 | `Model/no_folds/dataset_patches_no_fold.py` |
| NoduleNet3D, Models A and B, 3 seeds each | `Model/net.py`, `runs/cls_no_fold_mask_weighted/*/best.pt` |
| Suspicion threshold (Youden, chosen on val) | each run's `val_metrics.json`, averaged over the 3 seeds |

**Outputs per nodule:** where it is (slices, centre, size), YOLO11s's confidence,
and from Model B (soft labels, the system's answer) and Model A (hard labels, for comparison):

- the probability of each rating 1–5 (3 seeds averaged)
- the predicted rating and its confidence (the top probability)
- suspicion: P(4) + P(5), and whether it passes the threshold

One difference from training: the CNN's mask comes from YOLO's 2D boxes, because a
new scan has no radiologist outlines.

## Run it

```bash
# the website (loads both models once, then analyses scan after scan)
.venv/bin/python end_to_end/app.py
# open http://localhost:8000, then drop in a folder of DICOM files or a .zip

# or one scan from the command line
.venv/bin/python end_to_end/nodule_pipeline.py <folder or .zip> --out result.json
```

The server only listens on `localhost`, so scans stay on this computer. Nothing
beyond `requirements.txt` is needed: the server uses Python's standard library.

## Testing on NLST

NLST (National Lung Screening Trial) is a different set of patients from
LIDC-IDRI, so it shows how the system does on scans unlike anything it was
trained on. It has no nodule outlines like LIDC-IDRI's, so it tests that the
system runs end to end and what it finds, not its accuracy.

```bash
.venv/bin/python end_to_end/fetch_nlst.py --list 5                  # CT series and their sizes
.venv/bin/python end_to_end/fetch_nlst.py --series <SeriesInstanceUID>
.venv/bin/python end_to_end/nodule_pipeline.py end_to_end/nlst/<SeriesInstanceUID>.zip
```

Downloads go to `end_to_end/nlst/`, which git ignores.

## Limits

- Not a medical device. Ratings imitate LIDC-IDRI radiologists' opinions, not biopsy results.
- On the LIDC-IDRI test set YOLO11s found 87.6% of nodules (marked by ≥2 radiologists)
  at 1.8 false alarms per scan; nodules under 6 mm are missed most. Each false alarm
  is still rated by the CNN, which was only ever trained on real nodules.
- The end-to-end system as a whole has not been scored: that would need a dataset
  with both nodule locations and malignancy ratings that the models never saw.

# Classifying-malignancy-in-lung-nodules-using-deep-learning-models

## Decisions: 3D CNN, hard vs soft labels (no folds)

Fixed before scoring the test set. Nothing below changes after test results are seen.

### Research question
Does training on the spread of radiologists' ratings (soft labels, model B) give a better
malignancy classifier than training on a single median rating (hard labels, model A)?

### Data and labels
| Decision | Choice |
|---|---|
| Dataset | LIDC-IDRI CT scans; one XML file kept per series (the one with the most reading sessions) |
| Matching readers' nodules | Same nodule if centres are within ±10 px (x, y) and ±5 mm (z); at most one annotation per reader per nodule |
| `inclusion=FALSE` contours | Skipped: they are holes inside a nodule outline, so they never change the bounding box |
| Nodules used | 1,885 nodules marked by at least 2 readers (1-reader nodules excluded) |
| Hard label (model A) | Median reader rating, 1-5 |
| Rounding of tied medians | Halves rounded **up**: `math.floor(median + 0.5)`, so 3.5 → 4. Python's `round()` is not used because it rounds halves to the nearest even number. Affects 555 nodules (29%), each labelled one step higher |
| Soft label (model B) | Share of readers giving each rating 1-5 |
| Spread | Highest rating minus lowest rating; fixed before seeing any model output |

### Split
| Decision | Choice |
|---|---|
| Split | Single split by patient (`data/patient_split.csv`), stratified by each patient's most suspicious nodule: 1,305 train / 290 validation / 290 test nodules |
| Folds | Not used for this experiment (optional later) |
| YOLO | Same split as the YOLO detector, so both share the same test patients |

### Model input
| Decision | Choice |
|---|---|
| Saved patches | 48 mm cube at 1 mm voxels, centred on the nodule's 3D bounding box, cut from DICOM HU |
| Model input | 40 mm crop: centred for validation and test; shifted up to ±4 mm for training |
| HU scaling | Fixed window -1000 to 400 HU, mapped to -1 to 1 |
| Augmentation (training only) | Random flips on all 3 axes, random 0/90/180/270° turn in the axial plane, random shift. No rescaling (size is a key signal) |

### Model and training
| Decision | Choice |
|---|---|
| Network | `Model/net.py`: 4 conv blocks with batch norm, dropout 0.3, 291,861 weights. Identical for A and B |
| Loss | Cross-entropy for both (class index for A, reader shares for B) |
| Class imbalance | No class weights or oversampling; handled with imbalance-robust metrics |
| Optimiser | AdamW, learning rate 0.001, weight decay 0.0001, batch size 32 |
| Stopping | Early stopping on validation loss, patience 15, maximum 300 epochs; the best epoch is kept |
| Repeats | Seeds 1, 2, 3. A and B are paired by seed: same starting weights and batch order |
| Hardware | Google Colab, NVIDIA T4 GPU, for all six final runs |
| Script | `Model/no_folds/train_no_fold.py --labels {hard,soft} --seed {1,2,3} --epochs 300` |

### Evaluation
| Decision | Choice |
|---|---|
| Headline metrics | Balanced accuracy, AUC (suspicious vs not), cross-entropy vs readers, ECE |
| Extra metrics | Quadratic weighted kappa; sensitivity and specificity at a cut-off chosen on validation (Youden's J) |
| Not used | Macro F1 |
| "Suspicious" | Hard label (rounded median) of 4 or 5. Because tied medians round up, 169 of the 493 suspicious nodules are suspicious only through rounding (e.g. ratings `3;4`) |
| ECE | Measured against the median rating, as in `pipeline/metrics.py`; its effect on the soft-label model is discussed in the report |
| Baselines | Always predict 3; training-set label frequencies; diameter only (logistic regression on log diameter); one reader vs the median of the others. The last one describes reader disagreement and is not treated as a benchmark the model can beat |
| Confidence analysis | `confidence_by_spread`: mean top probability per level of reader disagreement, with spreads 3 and 4 grouped as "3+", plotted against the readers' own agreement |
| A vs B comparison | `bootstrap_diff`: 95% confidence interval for B - A, resampling whole patients |
| Test set | Scored once, after every decision above is fixed |

### Still to decide before the test set
- [ ] Cut-off for sensitivity / specificity: each seed's cut-off chosen on its own validation predictions, applied unchanged to its own test predictions?
- [ ] Combining the three seeds on test: score each seed and average, or pass all seeds to `bootstrap_diff`?
- [ ] Final runs: the six retrained 300-epoch runs only
- [ ] Figures made on test exactly as on validation: confidence by spread (with readers' agreement line), confusion matrices
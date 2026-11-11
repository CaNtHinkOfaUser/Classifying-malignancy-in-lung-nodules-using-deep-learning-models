# Setting up and running the project

This guide is for someone who has never seen the project before. It covers installing the Python
environment, checking that it works, and then the three things you can do with the project:

1. **Look at the results.** Re-score the test set and redraw the figures. Needs only this repository.
2. **Run the end-to-end system.** One CT scan in, every nodule found and rated. Needs only this
   repository and a CT scan.
3. **Reproduce everything from the raw data.** Download LIDC-IDRI, preprocess it and retrain both
   models. Needs about 130 GB of storage, and a GPU for training.

The trained weights and every data table are already in the repository, so levels 1 and 2 do not
need the 130 GB dataset.

---

## 1. What you need

| | Requirement |
|---|---|
| Python | **3.11** (the version the results were produced with). Check with `python3.11 --version` |
| Operating system | macOS 14 (Sonoma) or newer on Apple silicon (M1 to M4), Linux (x86-64 or ARM), or Windows 10/11 (64-bit) |
| Intel Mac | Not supported: PyTorch 2.14 has no Intel-Mac build. Use Google Colab (section 7) |
| Disk | About 2.5 GB for the environment and 0.4 GB for the repository. Level 3 needs about 130 GB more, ideally on an external SSD |
| GPU | Not needed for levels 1 and 2. For level 3, an NVIDIA GPU (CUDA) or an Apple-silicon Mac (MPS) trains the CNN; YOLO training needs an NVIDIA GPU (we used Google Colab and Kaggle) |
| Git | To clone the repository |

### Installing Python 3.11

- **macOS:** `brew install python@3.11` (with [Homebrew](https://brew.sh)), or the installer from
  [python.org](https://www.python.org/downloads/).
- **Windows:** the 3.11 installer from [python.org](https://www.python.org/downloads/windows/).
  Tick **"Add python.exe to PATH"**. Afterwards, `py -3.11 --version` should work.
- **Linux (Ubuntu/Debian):** `sudo apt install python3.11 python3.11-venv`.

---

## 2. Get the code

```bash
git clone https://github.com/CaNtHinkOfaUser/Classifying-malignancy-in-lung-nodules-using-deep-learning-models.git
```

```bash
cd Classifying-malignancy-in-lung-nodules-using-deep-learning-models
```

Every command below is run from this project folder unless it says otherwise.

---

## 3. Create the environment

The environment lives in a folder called `.venv` inside the project. Git ignores this folder, so
everyone makes their own.

### macOS and Linux

```bash
python3.11 -m venv .venv
```

```bash
.venv/bin/pip install --upgrade pip
```

```bash
.venv/bin/pip install -r requirements.txt
```

### Windows (PowerShell or Command Prompt)

```bat
py -3.11 -m venv .venv
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\pip install -r requirements.txt
```

On Windows, replace `.venv/bin/python` with `.venv\Scripts\python` in every command in this guide.

**NVIDIA GPU on Windows:** the plain `pip install` above gives a CPU-only PyTorch on Windows, which
is fine for levels 1 and 2. To train on the GPU, reinstall PyTorch with the command from the selector
at [pytorch.org/get-started](https://pytorch.org/get-started/locally/) (choose version 2.14.0, pip,
your CUDA version). On Linux, the standard install already includes CUDA support.

**Alternative: uv.** If you have [uv](https://docs.astral.sh/uv/), this is much faster:
`uv venv -p 3.11 .venv` then `uv pip install -r requirements.txt`.

### Check that it works

```bash
.venv/bin/python -c "import torch, ultralytics, pydicom, sklearn, pandas; print('torch', torch.__version__, '| cuda', torch.cuda.is_available(), '| mps', torch.backends.mps.is_available())"
```

You should see `torch 2.14.0`. `cuda True` (NVIDIA) or `mps True` (Apple silicon) means a GPU will be
used. If both are `False`, everything still runs on the CPU, just more slowly.

You don't need to activate the environment: calling `.venv/bin/python` uses it directly. If you
prefer to activate it, run `source .venv/bin/activate` (macOS/Linux) or `.venv\Scripts\activate`
(Windows), after which plain `python` works.

---

## 4. Notebooks (Jupyter kernel)

Most figures are made in notebooks in `Model/` and `General preprocessing/`. Register the environment
as a Jupyter kernel once:

```bash
.venv/bin/python -m ipykernel install --user --name lidc --display-name "LIDC (.venv)"
```

- **VS Code** (recommended): install the *Python* and *Jupyter* extensions, open the project folder,
  open a notebook, click **Select Kernel** at the top right and choose **LIDC (.venv)**.
- **Without VS Code:** `.venv/bin/pip install jupyterlab`, then `.venv/bin/jupyter lab`, and pick
  the **LIDC (.venv)** kernel.

Notebooks find the project folder from where they are opened, so open them from inside the project
(VS Code and Jupyter do this automatically).

---

## 5. Level 1: look at the results (no dataset needed)

The six final CNN runs (Models A and B, seeds 1 to 3, with the box mask and class-weighted loss) are
in `runs/cls_no_fold_mask_weighted/`, each with its `best.pt` weights and its saved validation and
test predictions. The YOLO weights are in `results/YOLO results/<model>/model/.../weights/best.pt`.

**Re-score the CNN test set** from the saved predictions (about a minute). It writes to a new
folder, so the reported results are not overwritten:

```bash
.venv/bin/python Model/no_folds/score_test_no_fold.py --runs-dir runs/cls_no_fold_mask_weighted --out results/rescored_check
```

`results/rescored_check/summary.csv` should match `results/test_no_fold_mask_redoweights/summary.csv`
(for example, AUC 0.957 for Model A and 0.958 for Model B).

**Redraw the figures** by opening these notebooks and choosing *Run All*:

| Notebook | Makes |
|---|---|
| `Model/confusion_matrices_test.ipynb` | the mean ± SD confusion matrices of Models A and B |
| `Model/reliability_test.ipynb` | the calibration (reliability) diagrams |
| `Model/predicted_ratings.ipynb` | predicted vs true rating counts |
| `General preprocessing/reader_agreement.ipynb`, `malignancy_distributions.ipynb` | how much the radiologists agree; the rating distribution |

Scripts for the remaining figures:

```bash
.venv/bin/python Model/cnn_poster/make_figures.py
```

```bash
.venv/bin/python Model/YOLO/yolo_confusion_cnn_style.py
```

---

## 6. Level 2: the end-to-end system (one CT scan in, rated nodules out)

This runs YOLO11s on every slice of a scan, joins its boxes into 3D nodules and rates each nodule
with Models A and B. See `end_to_end/README.md` for details.

```bash
.venv/bin/python end_to_end/app.py
```

Open <http://localhost:8765> and drop in a folder of a CT scan's DICOM files, or a `.zip` of them.
The server only listens on your own computer, so the scan never leaves it. Or, without the website:

```bash
.venv/bin/python end_to_end/nodule_pipeline.py <scan folder or .zip> --out result.json
```

You need a chest CT scan in DICOM format. To fetch a public one from the NLST dataset (no account
needed; `--list` shows each series' size first, and downloads go to `end_to_end/nlst/`):

```bash
.venv/bin/python end_to_end/fetch_nlst.py --list 5
```

```bash
.venv/bin/python end_to_end/fetch_nlst.py --series <SeriesInstanceUID from the list>
```

The website loads the seven models (YOLO11s and the six CNN runs) once at start-up, then analyses
scan after scan.

---

## 7. Level 3: reproduce everything from the raw data

### 7.1 Download LIDC-IDRI

1. Go to the LIDC-IDRI page on The Cancer Imaging Archive:
   <https://www.cancerimagingarchive.net/collection/lidc-idri/>.
2. Download the **images** (DICOM, about 125 GB). TCIA gives you a `.tcia` manifest file. Open it
   with the free **NBIA Data Retriever**, and choose the **Classic Directory Name** layout
   (`<Patient ID>/<Study UID>/<Series UID>/`), which is the layout the sorting scripts expect.
3. Download the **radiologists' annotations** (the XML zip, under *Radiologist Annotations/Segmentations*)
   and unzip it.
4. `data/metadata.csv` (the list of every series) is already in the repository.

We kept everything on an external drive named `Expansion`. The scripts expect:

| Path | Contents |
|---|---|
| `/Volumes/Expansion/lidc_idri/` | the DICOM download, one folder per patient |
| `/Volumes/Expansion/LIDC-XML/tcia-lidc-xml/` | the unzipped XML annotations |
| `/Volumes/Expansion/processed/` | everything the pipeline writes (patches, masks, YOLO images) |

**If your data is somewhere else**, change these paths in these places (they're near the top of each file):

- `pipeline/ct.py`, lines 30 and 31 (`DICOM_ROOT`, `PROCESSED`). Every later step reads them from here.
- `dicom_preproccessing/src/Sorting/ExtractCTScanToDirectFolder.py`, `DeleteDxFiles.py`,
  `dicom_preproccessing/src/RenamingDcm/RenamingDcm.py` (`dataset_folder`).
- `xml_preprocessing/xml_preprocessing_bb_a.py`, line 250 (`xml_folder_path`).

On Windows, use paths like `Path(r"D:\lidc_idri")`.

Some paths can also be set with environment variables instead of editing files: `LIDC_PATCH_DIR`,
`LIDC_MASK_DIR` (CNN patches and masks), `LIDC_PROCESSED` and `LIDC_YOLO_RUNS` (YOLO data and runs).

### 7.2 Run the pipeline, in order

Every step writes a file that the next one reads. The `data/*.csv` files in the repository are the
outputs of steps 2 to 6, so you can start from any step.

| # | Step | Command / notebook | Writes |
|---|---|---|---|
| 1 | Tidy the DICOM folders: move each CT series up into its patient folder, delete the X-ray (CR) studies, rename slices to `slice_XXXX.dcm` sorted by z | `.venv/bin/python dicom_preproccessing/src/Sorting/ExtractCTScanToDirectFolder.py`, then `DeleteDxFiles.py`, then `.venv/bin/python dicom_preproccessing/src/RenamingDcm/RenamingDcm.py` (the same steps are in `Sorting/DicomSorting.ipynb`) | the DICOM folders, in place. **Moves and deletes files: keep the original download until this works** |
| 2 | Read the radiologists' XML outlines, match the same nodule across readers | `.venv/bin/python xml_preprocessing/xml_preprocessing_bb_a.py` | `data/labels_bb_a.csv` |
| 3 | Hard and soft labels per nodule | `General preprocessing/label generation.ipynb` | `data/nodules_labels.csv` |
| 4 | Patient-level train / validation / test split | `General preprocessing/Patient_split.ipynb` | `data/patient_split.csv` |
| 5 | Index every CT slice (about 25 min over USB) | `.venv/bin/python pipeline/4abuild_slice_index.py` | `data/slice_index.csv`, `data/series_index.csv` |
| 6 | Cut a 48 mm cube and a box mask around every nodule (about 10 min) | `.venv/bin/python pipeline/4bbuild_patches.py` | `processed/patches_48mm/`, `processed/masks_48mm/` |
| 7 | Build the YOLO dataset (2.5D slice images and box labels) | `cd pipeline` then `../.venv/bin/python 5build_yolo_dataset.py` | `processed/yolo_lidc/` |
| 8 | Baselines (always-3, diameter only, one radiologist vs the rest) | `.venv/bin/python pipeline/6baselines.py` (add `--test` only at the very end) | `results/baselines_val_no_fold.csv` |

### 7.3 Train and test the 3D CNN (Models A and B)

Each run stops early, after roughly 40 to 80 epochs: about 25 to 50 minutes on an M2 Mac, or about
5 minutes on a Colab GPU. The final models are the class-weighted runs with the box mask, seeds 1 to 3:

```bash
for s in 1 2 3; do
  .venv/bin/python Model/no_folds/train_no_fold.py --labels hard --seed $s --epochs 300 --mask --loss weighted
  .venv/bin/python Model/no_folds/train_no_fold.py --labels soft --seed $s --epochs 300 --mask --loss weighted
done
```

To check the setup first, run `--epochs 1` (does it run?) and `--overfit 32` (can it memorise 32
nodules?). Runs are saved in `runs/cls_no_fold_mask_weighted/<hard|soft>_seed<k>/`.

The test set is predicted and scored **once**, after every decision is fixed:

```bash
.venv/bin/python Model/no_folds/predict_test_no_fold.py --runs-dir runs/cls_no_fold_mask_weighted
```

```bash
.venv/bin/python Model/no_folds/score_test_no_fold.py --runs-dir runs/cls_no_fold_mask_weighted --out results/test_no_fold_mask_redoweights
```

`predict_test_no_fold.py` refuses to overwrite existing test predictions unless you add `--force`.
This is deliberate, so that the test set can't be predicted twice by accident.

### 7.4 Train and test YOLO

YOLO needs an NVIDIA GPU. A 100-epoch run at 1024 px takes many hours, longer than one Colab
session, so the notebooks save to Google Drive and can resume (`yolo_kaggle_resume.ipynb`, `--resume`).
The scripts are in `Model/YOLO/yolo11s 1024 pixel code/pipeline/` (the final model) and are run from
that folder:

```bash
cd "Model/YOLO/yolo11s 1024 pixel code/pipeline"
```

```bash
../../../../.venv/bin/python pack_for_colab.py --jpeg
```

This makes about 3.9 GB of zips to upload to Google Drive. Then open
`../notebooks/yolo_colab.ipynb` in Google Colab with a GPU runtime and run it. It trains with:

```bash
python 6train_yolo.py --model yolo11s.pt --imgsz 1024
```

and, back on your computer with the dataset available, evaluates per nodule on every slice:

```bash
../../../../.venv/bin/python 7eval_yolo.py --weights <run folder>/weights/best.pt --split val
```

Use `--split test` only once, at the end. `yolo_val_test.py --weights ...` gives YOLO's own
precision, recall, mAP and plots.

### 7.5 Google Colab for the CNN

1. Zip `processed/patches_48mm/` and `processed/masks_48mm/`, upload them to Google Drive and unzip
   them in Colab.
2. Clone this repository in Colab and run `pip install -r requirements.txt`.
3. Set the folders with `%env LIDC_PATCH_DIR=/content/patches_48mm` and
   `%env LIDC_MASK_DIR=/content/masks_48mm`, then run the training commands in 7.3 with `python`
   in place of `.venv/bin/python`.

---

## 8. Troubleshooting

| Problem | Fix |
|---|---|
| `No matching distribution found for torch==2.14.0` | Wrong Python version (use 3.11), an Intel Mac, or macOS older than 14. Use Colab |
| `ModuleNotFoundError: No module named 'ct'` (or `net`, `metrics`) | Run the command from the project folder, or for the YOLO scripts from their own `pipeline/` folder, exactly as written above |
| `FileNotFoundError: /Volumes/Expansion/...` | That step needs the raw data. Set the paths as in 7.1 |
| The notebook kernel isn't listed | Run the `ipykernel install` command in section 4, then restart VS Code |
| `mps` errors on a Mac | `export PYTORCH_ENABLE_MPS_FALLBACK=1` and rerun, or add `--device cpu` for YOLO |
| Different numbers after retraining | GPU training is not bit-for-bit repeatable across machines; expect small differences between runs (about ±0.03 balanced accuracy between seeds), which is why every result is the mean of 3 seeds |

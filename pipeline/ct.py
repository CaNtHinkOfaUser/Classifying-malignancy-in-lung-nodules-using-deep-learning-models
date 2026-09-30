# Ishaan

"""Where the data lives, and the ONE way this project reads CT.

The patch cache, the YOLO images and the end-to-end demo all import from here,
so a nodule is cut out of a scan the same way at training time and at test
time. If you change how HU is read or how a cube is cut, change it here only.

Coordinate conventions (checked against the XML and DICOM headers):
    row  = y pixel index = XML yCoord = pixel_array[row, :]
    col  = x pixel index = XML xCoord = pixel_array[:, col]
    z    = ImagePositionPatient[2] in mm = XML imageZposition
    slice_XXXX.dcm files are sorted by z ASCENDING (RenamingDcm.py did this)
"""

# Emmaus notes
# Idk why the comments are so dramatic but anyways this code is just a module
# It contains the paths to all the important stuff
# It also contains a lot of CT functions to make navigating the dataset
# easier
# It also serves as configuration for certain stuff like output paths
# You can also configure other variables e.g. CUBE_MM

from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
from scipy.ndimage import map_coordinates

PROJECT = Path(__file__).resolve().parent.parent
DATA = PROJECT / "data"

DICOM_ROOT = Path("/Volumes/Expansion/lidc_idri")
PROCESSED = Path("/Volumes/Expansion/processed")
PATCH_DIR = PROCESSED / "patches_48mm"
YOLO_DIR = PROCESSED / "yolo_lidc"

METADATA = DATA / "metadata.csv"
BOXES = DATA / "labels_bb_a.csv"            # one row per nodule per slice, all readers
NODULES = DATA / "nodules_labels.csv"       # one row per nodule, >=2 readers, labels + split
SPLIT = DATA / "patient_split.csv"
SLICE_INDEX = DATA / "slice_index.csv"      # one row per DICOM file: series, idx, sop, z
SERIES_INDEX = DATA / "series_index.csv"    # one row per series: spacing, thickness, checks

AIR_HU = -1000                  # what goes where a cube runs off the edge of the scan
HU_MIN, HU_MAX = -1024, 3071    # the 12-bit CT range; below it is scanner padding
CUBE_MM = 48                    # cube side, in mm
CUBE_SPACING = 1.0              # mm per voxel in the cube -> 48 x 48 x 48
LUNG_WINDOW = (-1350, 150)      # level -600, width 1500: for YOLO images only


def series_dir(patient_id, series_uid):
    return DICOM_ROOT / patient_id / series_uid


def slice_path(patient_id, series_uid, idx):
    return series_dir(patient_id, series_uid) / f"slice_{idx:04d}.dcm"


def read_hu(path):
    """One slice as Hounsfield units, float32, clipped to the real CT range.

    float(), not int(): int() truncates a slope of 1.2 to 1.
    """
    ds = pydicom.dcmread(path)
    hu = ds.pixel_array.astype(np.float32) * float(ds.RescaleSlope) + float(ds.RescaleIntercept)
    return np.clip(hu, HU_MIN, HU_MAX)


def load_series_index():
    return pd.read_csv(SERIES_INDEX).set_index("series_instance_uid")


def load_slice_index():
    """slice_index.csv with at most one slice per (series, z).

    5 series (LIDC-IDRI-0085, -0146, -0418, -0572, -0979) store a second slice
    at a z that already has one. Interpolating along z needs strictly
    increasing positions, so the first file at each z is kept.
    """
    sl = pd.read_csv(SLICE_INDEX).sort_values(["series_instance_uid", "idx"])
    return sl.drop_duplicates(["series_instance_uid", "z"], keep="first").reset_index(drop=True)


def load_slab(patient_id, series_uid, slices, z_lo, z_hi):
    """Read every slice of one series whose z is in [z_lo, z_hi], plus one either side.

    slices: this series' rows of slice_index.csv, sorted by idx (so by z).
    Returns (volume float32 [n, rows, cols], z positions in mm).
    """
    zs = slices["z"].to_numpy()
    lo = max(int(np.searchsorted(zs, z_lo, side="left")) - 1, 0)
    hi = min(int(np.searchsorted(zs, z_hi, side="right")) + 1, len(zs))
    idxs = slices["idx"].to_numpy()[lo:hi]
    vol = np.stack([read_hu(slice_path(patient_id, series_uid, i)) for i in idxs])
    return vol, zs[lo:hi]


def extract_cube(vol, zs, row_spacing, col_spacing, centre_z, centre_row, centre_col,
                 size_mm=CUBE_MM, spacing=CUBE_SPACING):
    """Cut a cube of side size_mm around a point, resampled to `spacing` mm voxels.

    Resampling and cropping happen in one step: for every output voxel we work
    out where it sits in the original scan (fractional slice, row, col) and
    interpolate trilinearly. np.interp on the real z positions copes with
    uneven slice gaps; anything past the end of the scan becomes air.

    Returns int16 HU, shape (n, n, n) in (z, row, col) order, z ascending.
    """
    n = int(round(size_mm / spacing))
    offsets = (np.arange(n) - (n - 1) / 2) * spacing          # mm from the centre
    zi = np.interp(centre_z + offsets, zs, np.arange(len(zs)), left=-2.0, right=len(zs) + 1.0)
    ri = centre_row + offsets / row_spacing
    ci = centre_col + offsets / col_spacing
    grid = np.meshgrid(zi, ri, ci, indexing="ij")
    cube = map_coordinates(vol, grid, order=1, mode="constant", cval=AIR_HU)
    return np.round(cube).astype(np.int16)


def to_uint8_window(hu, window=LUNG_WINDOW):
    """Fixed HU -> 0..255 mapping. The same HU is always the same grey,
    on every slice of every scan (unlike a per-slice min/max stretch)."""
    lo, hi = window
    return (np.clip((hu - lo) / (hi - lo), 0, 1) * 255).round().astype(np.uint8)


def slice_rgb(vol, i):
    """2.5D image for YOLO: slices i-1, i, i+1 of `vol` as R, G, B.

    A nodule is a blob that appears and vanishes within a few slices; a vessel
    is a tube that carries on through all three. Stacking neighbours lets a 2D
    detector see that difference. The edge slice is repeated at the ends.
    """
    j = [max(i - 1, 0), i, min(i + 1, len(vol) - 1)]
    return np.stack([to_uint8_window(vol[k]) for k in j], axis=-1)

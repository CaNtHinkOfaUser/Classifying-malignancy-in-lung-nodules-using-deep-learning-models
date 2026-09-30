# Written by Claude for Ishaan

"""Step 4a. Read the header of every CT slice once and write down where it is.

Run after:  the DICOM sort/rename (slice_XXXX.dcm, z ascending).
Writes:     data/slice_index.csv   one row per DICOM file (series, idx, sop, z)
            data/series_index.csv  one row per series (spacing, thickness, checks)
Takes:      ~25 min over USB (header reads only, 16 threads).

Why this exists: a few LIDC XMLs point at SOP UIDs that aren't in the scan
folder at all (LIDC-IDRI-0017, -0365, -0659), although their z positions are
right. Every later step finds slices by (series, z) through this table instead
of by SOP UID, so those nodules still land on the right slice.
"""

import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import pydicom

from ct import DICOM_ROOT, METADATA, SERIES_INDEX, SLICE_INDEX

TAGS = ["SOPInstanceUID", "SeriesInstanceUID", "ImagePositionPatient",
        "ImageOrientationPatient", "PixelSpacing", "SliceThickness",
        "RescaleSlope", "RescaleIntercept", "Rows", "Columns"]


def read_header(job):
    patient_id, series_uid, name = job
    path = DICOM_ROOT / patient_id / series_uid / name
    h = pydicom.dcmread(path, stop_before_pixels=True, specific_tags=TAGS)
    row_sp, col_sp = (float(v) for v in h.PixelSpacing)
    return {
        "patient_id": patient_id,
        "series_instance_uid": series_uid,
        "idx": int(name[len("slice_"):-len(".dcm")]),
        "sop": str(h.SOPInstanceUID),
        "z": float(h.ImagePositionPatient[2]),
        "header_series_ok": str(h.SeriesInstanceUID) == series_uid,
        "axial": [round(float(v)) for v in h.ImageOrientationPatient] == [1, 0, 0, 0, 1, 0],
        "row_spacing": row_sp,
        "col_spacing": col_sp,
        "thickness": float(getattr(h, "SliceThickness", np.nan) or np.nan),
        "slope": float(h.RescaleSlope),
        "rows": int(h.Rows),
        "cols": int(h.Columns),
    }


if __name__ == "__main__":
    meta = pd.read_csv(METADATA)
    ct = meta[meta["Modality"] == "CT"].drop_duplicates("Series Instance UID")

    jobs, missing = [], []
    for pid, uid in zip(ct["Patient ID"], ct["Series Instance UID"]):
        folder = DICOM_ROOT / pid / uid
        if not folder.is_dir():
            missing.append((pid, uid))
            continue
        names = sorted(e.name for e in os.scandir(folder)
                       if e.name.startswith("slice_") and e.name.endswith(".dcm"))
        jobs += [(pid, uid, name) for name in names]
    print(f"{len(ct)} CT series in metadata, {len(missing)} folders missing, {len(jobs)} slices")

    t0, rows = time.time(), []
    with ThreadPoolExecutor(16) as pool:
        for k, r in enumerate(pool.map(read_header, jobs, chunksize=64)):
            rows.append(r)
            if k % 10000 == 0:
                print(f"  {k}/{len(jobs)}  {time.time() - t0:.0f}s", flush=True)

    sl = pd.DataFrame(rows).sort_values(["series_instance_uid", "idx"]).reset_index(drop=True)
    sl[["patient_id", "series_instance_uid", "idx", "sop", "z"]].to_csv(SLICE_INDEX, index=False)

    # ---- per-series summary and checks ----
    def summarise(g):
        dz = np.diff(g["z"].to_numpy())
        return pd.Series({
            "patient_id": g["patient_id"].iloc[0],
            "n_slices": len(g),
            "z_min": g["z"].min(),
            "z_max": g["z"].max(),
            "z_spacing": float(np.median(dz)) if len(dz) else np.nan,
            "z_ascending": bool((dz > 0).all()),        # file order == z order, no repeats
            "z_uniform": bool(len(dz) and np.abs(dz - np.median(dz)).max() < 0.05),
            "row_spacing": g["row_spacing"].iloc[0],
            "col_spacing": g["col_spacing"].iloc[0],
            "thickness": g["thickness"].median(),
            "rows": g["rows"].iloc[0],
            "cols": g["cols"].iloc[0],
            "axial": bool(g["axial"].all()),
            "header_series_ok": bool(g["header_series_ok"].all()),
            "one_spacing": g[["row_spacing", "col_spacing"]].nunique().max() == 1,
            "slope_is_1": bool((g["slope"] == 1).all()),
        })

    se = sl.groupby("series_instance_uid").apply(summarise, include_groups=False).reset_index()
    se.to_csv(SERIES_INDEX, index=False)

    print(f"\nFinished in {time.time() - t0:.0f}s: {len(sl)} slices, {len(se)} series")
    print(f"  -> {SLICE_INDEX}\n  -> {SERIES_INDEX}")
    for col in ["z_ascending", "z_uniform", "axial", "header_series_ok", "one_spacing", "slope_is_1"]:
        print(f"series failing {col:17s}: {int((~se[col].astype(bool)).sum())}")
    print(f"pixel spacing {se.row_spacing.min():.3f}-{se.row_spacing.max():.3f} mm, "
          f"slice spacing {se.z_spacing.min():.2f}-{se.z_spacing.max():.2f} mm, "
          f"image size {sorted(se.rows.unique())} x {sorted(se.cols.unique())}")

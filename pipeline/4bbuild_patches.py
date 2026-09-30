# Ishaan

"""Step 4b. Cut one 48 mm cube around every nodule and save it.

Run after:  build_slice_index.py, label generation, split_cell.py
Reads:      data/nodules_labels.csv   the 1,885 nodules (>=2 readers) with labels + split
            data/labels_bb_a.csv      every slice each nodule was outlined on
            data/slice_index.csv      (series, z) -> file
Writes:     /Volumes/Expansion/processed/patches_48mm/<nodule_key>.npy
                int16 HU, shape (48, 48, 48), axes (z, row, col), 1 mm voxels, z ascending
            extra columns in data/nodules_labels.csv (nodule_key, patch_file, geometry, QC)
Takes:      ~10 min over USB.

The cube is centred on the middle of the nodule's 3D bounding box -- the box
around every outline on every slice -- not on the widest slice, so a nodule
that is lopsided in z still sits in the middle of its cube.
"""

# Emmaus notes
# Naming convention of the numpy files is
# Patient ID_Last 6 digits of seriesinstanceuid_merged nodule id with a zero at the start if it is single digit
# This code takes 30 minutes to run
# Note that the patches are not sorted into their respective train test val folders
# Refer to patient_split.csv for the splits as the source of truth


import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from ct import (BOXES, CUBE_MM, CUBE_SPACING, NODULES, PATCH_DIR,
                extract_cube, load_series_index, load_slice_index, read_hu, slice_path)

KEY = ["patient_id", "series_instance_uid", "merged_nodule_id"]
Z_MATCH_MM = 0.1          # an outline's z must be this close to a real slice
NEW_COLS = ["nodule_key", "patch_file", "centre_z_mm", "centre_row_px", "centre_col_px",
            "pixel_spacing_mm", "slice_spacing_mm", "diameter_mm",
            "qc_core_max_hu", "qc_ring_hu", "qc_pad_frac"]

# QC probes: distance of every voxel from the cube centre, in mm
_off = (np.arange(CUBE_MM) - (CUBE_MM - 1) / 2) * CUBE_SPACING
_R = np.sqrt(_off[:, None, None] ** 2 + _off[None, :, None] ** 2 + _off[None, None, :] ** 2)
CORE = _R <= 3                      # within 3 mm of the centre
RING = (_R >= 18) & (_R <= 22)      # a shell 20 mm out: mostly lung


def nodule_key(row):
    # patient + last 6 digits of the series (7 patients have two scans) + nodule number
    return f"{row.patient_id}_{row.series_instance_uid[-6:]}_{int(row.merged_nodule_id):02d}"


def read_slab(patient_id, series_uid, idxs, cache, pool):
    """HU for the given slice numbers, reusing slices already read for this series."""
    todo = [i for i in idxs if i not in cache]
    paths = [slice_path(patient_id, series_uid, i) for i in todo]
    for i, hu in zip(todo, pool.map(read_hu, paths)):
        cache[i] = hu
    return np.stack([cache[i] for i in idxs])


if __name__ == "__main__":
    PATCH_DIR.mkdir(parents=True, exist_ok=True)

    nodules = pd.read_csv(NODULES)
    nodules = nodules.drop(columns=[c for c in NEW_COLS if c in nodules.columns])
    nodules["nodule_key"] = nodules.apply(nodule_key, axis=1)
    assert nodules["nodule_key"].is_unique, "nodule_key collision"

    boxes = pd.read_csv(BOXES).merge(nodules[KEY + ["nodule_key"]], on=KEY)
    boxes["x_min"] = boxes["x_centre"] - boxes["width"] / 2
    boxes["x_max"] = boxes["x_centre"] + boxes["width"] / 2
    boxes["y_min"] = boxes["y_centre"] - boxes["height"] / 2
    boxes["y_max"] = boxes["y_centre"] + boxes["height"] / 2

    series = load_series_index()
    slices = load_slice_index().sort_values(["series_instance_uid", "idx"])
    slices_by_series = dict(tuple(slices.groupby("series_instance_uid")))

    # ---- geometry per nodule, every slice resolved by z, not by SOP UID ----
    geo = {}
    z_errors = []
    for nk, g in boxes.groupby("nodule_key"):
        uid = g["series_instance_uid"].iloc[0]
        zs = slices_by_series[uid]["z"].to_numpy()
        err = np.abs(zs[None, :] - g["z-slice"].to_numpy()[:, None]).min(axis=1)
        z_errors.append(err.max())
        s = series.loc[uid]
        geo[nk] = {
            "centre_z_mm": (g["z-slice"].min() + g["z-slice"].max()) / 2,
            "centre_row_px": (g["y_min"].min() + g["y_max"].max()) / 2,
            "centre_col_px": (g["x_min"].min() + g["x_max"].max()) / 2,
            "pixel_spacing_mm": s["col_spacing"],
            "slice_spacing_mm": s["z_spacing"],
            "n_slices": g["z-slice"].nunique(),     # was double-counted on LIDC-IDRI-0017
            "extent_z_mm": g["z-slice"].max() - g["z-slice"].min() + s["z_spacing"],
            "extent_xy_mm": max(g["x_max"].max() - g["x_min"].min(),
                                g["y_max"].max() - g["y_min"].min()) * s["col_spacing"],
        }
    z_errors = np.array(z_errors)
    print(f"outline z vs nearest real slice: max error {z_errors.max():.4f} mm; "
          f"nodules over {Z_MATCH_MM} mm: {(z_errors > Z_MATCH_MM).sum()} (MUST be 0)")
    assert (z_errors <= Z_MATCH_MM).all()

    geo = pd.DataFrame.from_dict(geo, orient="index")
    nodules = nodules.drop(columns=["n_slices"]).merge(
        geo, left_on="nodule_key", right_index=True, how="left")
    nodules["diameter_mm"] = nodules["diameter_px"] * nodules["pixel_spacing_mm"]
    nodules["patch_file"] = nodules["nodule_key"] + ".npy"

    # ---- cut the cubes, one series at a time so each slice is read once ----
    half = CUBE_MM / 2 + 1
    qc = {}
    t0 = time.time()
    with ThreadPoolExecutor(8) as pool:
        for k, (uid, group) in enumerate(nodules.groupby("series_instance_uid")):
            sl = slices_by_series[uid]
            zs_all, idx_all = sl["z"].to_numpy(), sl["idx"].to_numpy()
            s = series.loc[uid]
            pid = group["patient_id"].iloc[0]
            cache = {}
            for row in group.itertuples():
                zc, rc, cc = row.centre_z_mm, row.centre_row_px, row.centre_col_px
                # every slice within the cube's z range, plus one either side
                lo = max(int(np.searchsorted(zs_all, zc - half)) - 1, 0)
                hi = min(int(np.searchsorted(zs_all, zc + half, side="right")) + 1, len(zs_all))
                vol = read_slab(pid, uid, idx_all[lo:hi], cache, pool)
                zs = zs_all[lo:hi]

                cube = extract_cube(vol, zs, s["row_spacing"], s["col_spacing"], zc, rc, cc)
                np.save(PATCH_DIR / row.patch_file, cube)

                # QC: share of the cube that fell off the scan, and how dense the centre is
                off = (np.arange(CUBE_MM) - (CUBE_MM - 1) / 2) * CUBE_SPACING
                in_z = ((zc + off >= s["z_min"]) & (zc + off <= s["z_max"])).mean()
                in_r = ((rc + off / s["row_spacing"] >= 0) & (rc + off / s["row_spacing"] <= s["rows"] - 1)).mean()
                in_c = ((cc + off / s["col_spacing"] >= 0) & (cc + off / s["col_spacing"] <= s["cols"] - 1)).mean()
                qc[row.nodule_key] = {
                    # densest voxel within 3 mm of the centre: soft tissue (~0 HU) for a
                    # solid nodule, -700..-300 for ground glass, air for a cavity
                    "qc_core_max_hu": float(cube[CORE].max()),
                    "qc_ring_hu": float(np.median(cube[RING])),
                    "qc_pad_frac": round(1 - in_z * in_r * in_c, 4),
                }
            if k % 50 == 0:
                print(f"  series {k}  nodules {len(qc)}/{len(nodules)}  {time.time() - t0:.0f}s", flush=True)

    qc = pd.DataFrame.from_dict(qc, orient="index")
    nodules = nodules.merge(qc[["qc_core_max_hu", "qc_ring_hu", "qc_pad_frac"]], left_on="nodule_key",
                            right_index=True, how="left")

    # keep the original columns in their order, new ones at the end
    front = [c for c in nodules.columns if c not in NEW_COLS + ["extent_z_mm", "extent_xy_mm"]]
    out = nodules[front + NEW_COLS]
    out.to_csv(NODULES, index=False)

    # ---- checkpoints ----
    files = sorted(p.name for p in PATCH_DIR.glob("*.npy") if not p.name.startswith("."))   # skip macOS ._ sidecars
    print(f"\nFinished in {time.time() - t0:.0f}s: {len(qc)} cubes -> {PATCH_DIR}")
    print(f"files on disk: {len(files)}  (MUST equal {len(nodules)})")
    print(f"every nodule has a cube: {set(nodules.patch_file) <= set(files)}")
    test = np.load(PATCH_DIR / nodules.patch_file.iloc[0])
    print(f"one cube: shape {test.shape}, dtype {test.dtype}, HU {test.min()}..{test.max()}")

    print("\nis the nodule at the centre? (lung is about -850 HU)") # ??? Lungs is -600 HU
    for col in ["qc_core_max_hu", "qc_ring_hu"]:
        q = qc[col].quantile([0.1, 0.5, 0.9]).round(0).tolist()
        print(f"  {col:15s} p10 {q[0]:6.0f}  median {q[1]:6.0f}  p90 {q[2]:6.0f}")
    print(f"  core denser than the ring 20 mm out: {(qc.qc_core_max_hu > qc.qc_ring_hu).mean():.1%}")
    print(f"  nothing solid within 3 mm (< -600 HU): {(qc.qc_core_max_hu < -600).sum()} "
          "-- LOOK at these: on 25 Sep all 19 were ground-glass or cavitary, correctly centred")
    print(f"cubes running off the scan edge: {(qc.qc_pad_frac > 0).sum()} "
          f"(max {qc.qc_pad_frac.max():.1%} of the cube)")
    print(f"nodules bigger than the cube in some axis: "
          f"{int(((geo.extent_z_mm > CUBE_MM) | (geo.extent_xy_mm > CUBE_MM)).sum())}")
    print(f"diameter_mm: median {nodules.diameter_mm.median():.1f}, "
          f"p95 {nodules.diameter_mm.quantile(.95):.1f}, max {nodules.diameter_mm.max():.1f}")
    print(f"extent_xy_mm p99 {geo.extent_xy_mm.quantile(.99):.1f}, extent_z_mm p99 {geo.extent_z_mm.quantile(.99):.1f}")

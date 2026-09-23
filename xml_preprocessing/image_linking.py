import pydicom
import pandas as pd
from pathlib import Path

DCM_ROOT = Path("/Volumes/Expansion/lidc_idri")
PNG_ROOT = Path("/Volumes/Expansion/LIDC-IDRI")
project_folder = Path(__file__).parent.parent
labels_path = project_folder / "data" / "labels_bb_a.csv"
labels = pd.read_csv(labels_path)

rows = []
for i, dcm_path in enumerate(DCM_ROOT.rglob("*.dcm")):
    ds = pydicom.dcmread(dcm_path, stop_before_pixels=True)
    if ds.get("Modality") != "CT":
        continue
    png_path = PNG_ROOT / dcm_path.relative_to(DCM_ROOT).with_suffix(".png")
    rows.append({
        "image_sop_id": ds.SOPInstanceUID,
        "png_path": png_path.as_posix(),
        "img_width": int(ds.Columns),
        "img_height": int(ds.Rows),
    })

    if i % 50 == 0:
        print(i)

lookup = pd.DataFrame(rows)
lookup.to_csv("sop_to_png.csv", index=False)

lookup = pd.read_csv("sop_to_png.csv")

labels = labels.merge(lookup, on="image_sop_id", how="left")
print("rows with no PNG:", labels["png_path"].isna().sum())
labels.to_csv("labels_with_paths.csv", index=False)
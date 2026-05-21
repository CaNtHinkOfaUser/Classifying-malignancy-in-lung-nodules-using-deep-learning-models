from pathlib import Path
import pydicom
import pandas as pd

print("Start")

project_folder = Path(__file__).parent.parent.parent
dataset_folder = Path("/Volumes/Expansion/lidc_idri")
metadata = project_folder / "data" / "metadata.csv"
df = pd.read_csv(metadata)

for index, row in df.iterrows():
    if row["Modality"] == "CT":
        patient_id = row["Patient ID"]
        patient_folder = dataset_folder / patient_id
        series_instance_uid = row["Series Instance UID"]
        ct_folder = patient_folder / series_instance_uid
        
        series_dict = {}

        for file in ct_folder.rglob("*.dcm"):
            dicom = pydicom.dcmread(file)
            series_id = dicom.SeriesInstanceUID

            if series_id not in series_dict:
                series_dict[series_id] = []
            series_dict[series_id].append((file, dicom))

        for series_id, slices in series_dict.items():
            slices.sort(key=lambda x: float(x[1].ImagePositionPatient[2]))

            for i, (file, dicom) in enumerate(slices):
                new_name = file.parent / f"slice_{i:04d}.dcm"
                file.rename(new_name)
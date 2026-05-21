# Emmaus

from pathlib import Path
import pandas as pd
import shutil

project_folder = Path(__file__).parent.parent.parent
dataset_folder = Path("/Volumes/Expansion/lidc_idri")
metadata = project_folder / "data" / "metadata.csv"
df = pd.read_csv(metadata)

deleted = 0
missing = 0

for index, row in df.iterrows():
    if row["Modality"] == "CR":
        study_instance_uid = row["Study Instance UID"]

        patient_id = row["Patient ID"]
        patient_folder = dataset_folder / patient_id
        dx_folder = patient_folder / study_instance_uid

        if dx_folder.exists():
            print(f"Deleting {row["Modality"]}: {dx_folder}")
            shutil.rmtree(dx_folder)
            deleted += 1
        else:
            missing += 1

print("\nFinished")
print(f"Deleted series: {deleted}")
print(f"Missing series: {missing}")

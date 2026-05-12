from pathlib import Path
import pandas as pd
import shutil

project_folder = Path(__file__).parent.parent
dataset_folder = Path("/Volumes/Expansion/lidc_idri")
metadata = project_folder / "data" / "metadata.csv"

df = pd.read_csv(metadata)

for index, row in df.iterrows():
    if row["Modality"] == "CT":
        patient_id = row["Patient ID"]
        patient_folder = dataset_folder / patient_id
        study_instance_uid = row["Study Instance UID"]
        series_instance_uid = row["Series Instance UID"]
        ct_folder = patient_folder / study_instance_uid / series_instance_uid

        shutil.move(ct_folder, patient_folder)
        shutil.rmtree(dataset_folder / patient_id / study_instance_uid)
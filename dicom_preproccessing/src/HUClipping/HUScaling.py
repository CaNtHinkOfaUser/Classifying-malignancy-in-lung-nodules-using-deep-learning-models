# Emmaus

from pathlib import Path 
import numpy as np
import pydicom
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from PIL import Image

def rescale_to_hu(dicom):
    img = dicom.pixel_array.astype(np.int32)
    img = int(dicom.RescaleSlope) * img + int(dicom.RescaleIntercept)
    return img

def apply_window(img: np.ndarray, window_level=-600, window_width=1500):
    upper = window_level + window_width // 2 
    lower = window_level - window_width // 2
    windowed_img = np.clip(img.copy(), lower, upper)
    windowed_img = windowed_img - np.min(windowed_img)
    windowed_img = windowed_img / np.max(windowed_img)
    windowed_img = (windowed_img * 255.0).astype('uint8')
    return windowed_img

project_folder = Path(__file__).parent.parent.parent
dataset_folder = project_folder / "lidc_idri" # to be changed
new_dataset_folder = project_folder / "LIDC-IDRI" # to be changed
metadata = project_folder / "data" / "metadata.csv"
df = pd.read_csv(metadata)

print("Start")

for index, row in df.iterrows():
    if row["Modality"] == "CT" and row["Patient ID"] == "LIDC-IDRI-0001": # to be changed
        patient_id = row["Patient ID"]
        patient_folder = dataset_folder / patient_id
        series_instance_uid = row["Series Instance UID"]
        input_folder = patient_folder / series_instance_uid
        output_folder = new_dataset_folder / patient_id / series_instance_uid
        output_folder.mkdir(parents=True, exist_ok=True)

        for file in input_folder.iterdir():
            if file.suffix == ".dcm":
                dicom = pydicom.dcmread(file)
                img = rescale_to_hu(dicom)
                img = apply_window(img)
                
                Image.fromarray(img).save(output_folder / f"{file.stem}.png")
    
    if index % 10:
        print(index)


print("Finish")
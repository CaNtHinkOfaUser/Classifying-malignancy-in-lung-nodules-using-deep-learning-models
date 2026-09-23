# Jairus and Emmaus

# This version creates a csv file containing the bounding boxes instead of roi annotations

import xml.etree.ElementTree as ET
import pandas as pd
from pathlib import Path

def get_xml_files(xml_folder_path):
    xml_files = []
    for folder in Path.iterdir(xml_folder_path):
        folder_path = xml_folder_path / folder

        if str(folder).startswith(".") or not Path.is_dir(folder_path):
            continue

        for f in Path.iterdir(folder_path):
            if str(f).startswith(".") or not str(f).endswith(".xml"):
                continue

            path = folder_path / f
            xml_files.append(path)
    return xml_files

def get_meta_data(data_header: str, meta_data_path: str, instance_uid: str):
    df = pd.read_csv(meta_data_path)
    target_patient_row = df[df["Series Instance UID"] == instance_uid]
    target_data = target_patient_row[data_header]
    data = str(target_data.iloc[0])
    return data

def parse_and_add_xml(xml_path, meta_data_path, all_rois_all_series, seen):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    ns = {"lidc": root.tag.split('}')[0].strip('{')}

    series_instance_uid_tag = root.find(".//lidc:SeriesInstanceUid", ns)
    instance_uid = ""
    if series_instance_uid_tag is None:
        series_instance_uid_tag = root.find(".//lidc:CTSeriesInstanceUid", ns)
        instance_uid = series_instance_uid_tag.text.strip()
    else:
        instance_uid = series_instance_uid_tag.text.strip()
    
    if not get_meta_data("Modality", meta_data_path, instance_uid) == "CT":
        return
    patient_id = get_meta_data("Patient ID", meta_data_path, instance_uid)
    
    found_nodules = root.findall(".//lidc:unblindedReadNodule", ns)
    for nodule in found_nodules:
        nodule_id_tag = nodule.find(".//lidc:noduleID", ns)
        if nodule_id_tag is None:
            continue
        nodule_id = nodule_id_tag.text.strip()

        malignancy_tag = nodule.find(".//lidc:malignancy", ns)
        if malignancy_tag is None:
            continue
        malignancy = malignancy_tag.text.strip()

        rois = nodule.findall(".//lidc:roi", ns)
        for roi in rois:
            xml_z = float(roi.find(".//lidc:imageZposition", ns).text.strip())
            image_sop_id = roi.find(".//lidc:imageSOP_UID", ns).text.strip()
            edge_maps = roi.findall(".//lidc:edgeMap", ns)
            contour = [(int(p.find(".//lidc:xCoord", ns).text), int(p.find(".//lidc:yCoord", ns).text)) for p in edge_maps]
            xs = [p[0] for p in contour]
            ys = [p[1] for p in contour]

            width = max(xs) - min(xs)
            height = max(ys) - min(ys)
            x_centre = (min(xs) + max(xs)) / 2
            y_centre = (min(ys) + max(ys)) / 2

            nodule_slice_dict = {
                "patient_id": patient_id,
                "series_instance_uid": instance_uid,
                "nodule_id": nodule_id,
                "image_sop_id": image_sop_id,
                "z-slice": xml_z,
                "width": width,
                "height": height,
                "x_centre": x_centre,
                "y_centre": y_centre,
                "malignancy": malignancy
            }

            duplicate_key = (
                nodule_slice_dict["patient_id"],
                nodule_slice_dict["series_instance_uid"],
                nodule_slice_dict["nodule_id"],
                nodule_slice_dict["image_sop_id"],
                nodule_slice_dict["z-slice"],
                nodule_slice_dict["width"],
                nodule_slice_dict["height"],
                nodule_slice_dict["x_centre"],
                nodule_slice_dict["y_centre"],
                nodule_slice_dict["malignancy"]
            )

            if duplicate_key in seen:
                continue

            seen.add(duplicate_key)
            all_rois_all_series.append(nodule_slice_dict)





if __name__ == "__main__":
    ROOT = Path(__file__).parent / "tcia-lidc-xml"
    meta_data_path = ROOT.parent / "metadata.csv"

    xml_files = get_xml_files(ROOT)

    all_rois_all_series = []
    seen = set()

    for i, xml_path in enumerate(xml_files):
        parse_and_add_xml(xml_path, meta_data_path, all_rois_all_series, seen)
        if i % 50 == 0:
            print(i)

    all_rois_all_series.sort(key=lambda d: (d["patient_id"], d["image_sop_id"]))
    df = pd.DataFrame(all_rois_all_series)
    df.to_csv("labels_bb.csv", index=False)

    print("Finished creating csv file")

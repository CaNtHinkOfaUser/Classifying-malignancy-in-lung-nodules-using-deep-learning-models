# Jairus and Emmaus
# This version creates a csv file which contains the roi annotations

import xml.etree.ElementTree as ET
import pandas as pd
from pathlib import Path

def get_xml_files(xml_folder_path):
    return [
        path for path in Path(xml_folder_path).glob("*/*.xml")
        if not path.name.startswith(".")
    ]

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

            nodule_slice_dict = {
                "patient_id": patient_id,
                "series_instance_uid": instance_uid,
                "nodule_id": nodule_id,
                "image_sop_id": image_sop_id,
                "z-slice": xml_z,
                "xs": xs,
                "ys": ys,
                "malignancy": malignancy
            }

            duplicate_key = (
                nodule_slice_dict["patient_id"],
                nodule_slice_dict["series_instance_uid"],
                nodule_slice_dict["nodule_id"],
                nodule_slice_dict["image_sop_id"],
                nodule_slice_dict["z-slice"],
                tuple(nodule_slice_dict["xs"]),
                tuple(nodule_slice_dict["ys"]),
                nodule_slice_dict["malignancy"]
            )

            if duplicate_key in seen:
                return

            seen.add(duplicate_key)
            all_rois_all_series.append(nodule_slice_dict)





if __name__ == "__main__":
    xml_folder_path = Path("/Volumes/Expansion/LIDC-XML/tcia-lidc-xml")
    project_folder = Path(__file__).parent.parent
    meta_data_path = project_folder / "data" / "metadata.csv"

    xml_files = get_xml_files(xml_folder_path)

    all_rois_all_series = []
    seen = set()

    for i, xml_path in enumerate(xml_files):
        try:
            parse_and_add_xml(xml_path, meta_data_path, all_rois_all_series, seen)
        except ET.ParseError as e:
            print(f"ERROR in: {xml_path}")
            print(e)
            break
        if i % 50 == 0:
            print(i)

    all_rois_all_series.sort(key=lambda d: (d["patient_id"], d["image_sop_id"]))
    df = pd.DataFrame(all_rois_all_series)
    df.to_csv(project_folder / "data" / "labels_roi.csv", index=False)

    print("Finished creating csv file")

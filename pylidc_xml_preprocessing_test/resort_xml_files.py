from pathlib import Path
import pandas as pd
import shutil
import xml.etree.ElementTree as ET


def get_xml_files(xml_folder_path):
    return [
        path for path in Path(xml_folder_path).glob("*/*.xml")
        if not path.name.startswith(".")
    ]


def load_meta_data(meta_data_path):
    df = pd.read_csv(meta_data_path)

    df = df.drop_duplicates("Series Instance UID")

    return df.set_index("Series Instance UID")[
        ["Modality", "Patient ID", "Study Instance UID"]
    ].to_dict("index")


def get_series_instance_uid(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    ns = {"lidc": root.tag.split("}")[0].strip("{")}

    series_instance_uid_tag = root.find(".//lidc:SeriesInstanceUid", ns)

    if series_instance_uid_tag is None:
        series_instance_uid_tag = root.find(".//lidc:CTSeriesInstanceUid", ns)

    if series_instance_uid_tag is not None and series_instance_uid_tag.text:
        return series_instance_uid_tag.text.strip()

    return None


def reorganise_xml_files(xml_folder_path, meta_data_path, output_folder):
    xml_folder_path = Path(xml_folder_path)
    output_folder = Path(output_folder)

    metadata = load_meta_data(meta_data_path)
    xml_files = get_xml_files(xml_folder_path)

    matched = 0
    unmatched = 0
    duplicates = 0

    xml_by_series = {}

    for xml_path in xml_files:

        series_uid = get_series_instance_uid(xml_path)

        if series_uid is None:
            print(f"Warning: No SeriesInstanceUID found in {xml_path}")
            continue

        if series_uid in xml_by_series:
            print(
                f"Warning: Multiple XML files found for "
                f"SeriesInstanceUID {series_uid}:"
            )
            print(f"  Existing: {xml_by_series[series_uid]}")
            print(f"  New:      {xml_path}")

            duplicates += 1
            continue

        xml_by_series[series_uid] = xml_path

    # Metadata is source of truth
    for series_uid, info in metadata.items():
        patient_id = str(info["Patient ID"])
        study_uid = str(info["Study Instance UID"])
        modality = str(info["Modality"])

        xml_path = xml_by_series.get(series_uid)

        if xml_path is None:
            print(
                f"WARNING: No XML found for "
                f"SeriesInstanceUid {series_uid}"
            )
            unmatched += 1
            continue

        destination = (
            output_folder
            / patient_id
            / study_uid
            / series_uid
        )

        destination.mkdir(parents=True, exist_ok=True)

        destination_file = destination / xml_path.name

        shutil.copy2(xml_path, destination_file)

        matched += 1

        print(
            f"[{modality}] "
            f"{patient_id} / {study_uid} / {series_uid}"
        )

    print("\nFinished.")
    print(f"Matched XML files:    {matched}")
    print(f"Unmatched series:      {unmatched}")
    print(f"Duplicate XML series:  {duplicates}")

project_folder = Path(__file__).parent.parent
meta_data_path = project_folder / "data" / "metadata.csv"

reorganise_xml_files(
    xml_folder_path="/Volumes/Expansion/LIDC-XML/tcia-lidc-xml",
    meta_data_path=meta_data_path,
    output_folder="/Volumes/Expansion/LIDC_XML_PyLIDC"
)
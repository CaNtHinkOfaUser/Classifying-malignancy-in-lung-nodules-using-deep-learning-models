# Jairus and Emmaus

# This version creates a csv file containing the bounding boxes instead of roi annotations
# It also blends the malignancy ratings together for YOLO model A, where the average rating is used

import xml.etree.ElementTree as ET
import pandas as pd
from pathlib import Path

XY_TOLERANCE_PX = 10.0   
Z_TOLERANCE_MM = 5.0
ALLOW_SAME_READER_MERGE = False
MERGE_SLICES = True

OUTPUT_CSV = "labels_bb_a.csv"

def get_xml_files(xml_folder_path):
    xml_files = []
    for folder in xml_folder_path.iterdir():
        if folder.name.startswith(".") or not folder.is_dir():
            continue
        for f in folder.iterdir():
            if f.name.startswith(".") or f.suffix != ".xml":
                continue
            xml_files.append(f)
    return xml_files


def load_meta_data(meta_data_path):
    df = pd.read_csv(meta_data_path)
    df = df.drop_duplicates("Series Instance UID")
    return df.set_index("Series Instance UID")[["Modality", "Patient ID"]].to_dict("index")

def parse_xml(xml_path, meta):
    root = ET.parse(xml_path).getroot()
    ns = {"lidc": root.tag.split("}")[0].strip("{")}

    uid_tag = root.find(".//lidc:SeriesInstanceUid", ns)
    if uid_tag is None:
        uid_tag = root.find(".//lidc:CTSeriesInstanceUid", ns)
    if uid_tag is None:
        return []
    uid = uid_tag.text.strip()

    info = meta.get(uid)
    if info is None or info["Modality"] != "CT":
        return []
    patient_id = str(info["Patient ID"])

    nodules = []
    sessions = root.findall(".//lidc:readingSession", ns)
    for reader_idx, session in enumerate(sessions):
        reader = f"{xml_path.stem}_r{reader_idx}"

        for nodule in session.findall("lidc:unblindedReadNodule", ns):
            nodule_id_tag = nodule.find("lidc:noduleID", ns)
            malignancy_tag = nodule.find(".//lidc:malignancy", ns)
            if nodule_id_tag is None or malignancy_tag is None:
                continue

            slices = []
            seen_slices = set()
            for roi in nodule.findall("lidc:roi", ns):
                inclusion = roi.find("lidc:inclusion", ns)
                if inclusion is not None and inclusion.text.strip().upper() == "FALSE":
                    continue

                edge_maps = roi.findall("lidc:edgeMap", ns)
                if not edge_maps:
                    continue
                xs = [int(p.find("lidc:xCoord", ns).text) for p in edge_maps]
                ys = [int(p.find("lidc:yCoord", ns).text) for p in edge_maps]

                sop = roi.find("lidc:imageSOP_UID", ns).text.strip()
                z = float(roi.find("lidc:imageZposition", ns).text.strip())
                box = (min(xs), min(ys), max(xs), max(ys))

                key = (sop, box)
                if key in seen_slices:
                    continue
                seen_slices.add(key)

                slices.append({
                    "image_sop_id": sop,
                    "z": z,
                    "x_min": box[0], "y_min": box[1],
                    "x_max": box[2], "y_max": box[3],
                })

            if not slices:
                continue

            n = len(slices)
            nodules.append({
                "patient_id": patient_id,
                "series_instance_uid": uid,
                "reader": reader,
                "nodule_id": nodule_id_tag.text.strip(),
                "malignancy": float(malignancy_tag.text.strip()),
                "slices": slices,
                "cx": sum((s["x_min"] + s["x_max"]) / 2 for s in slices) / n,
                "cy": sum((s["y_min"] + s["y_max"]) / 2 for s in slices) / n,
                "cz": sum(s["z"] for s in slices) / n,
            })
    return nodules


def is_close(a, b):
    return (abs(a["cx"] - b["cx"]) <= XY_TOLERANCE_PX
            and abs(a["cy"] - b["cy"]) <= XY_TOLERANCE_PX
            and abs(a["cz"] - b["cz"]) <= Z_TOLERANCE_MM)


def scaled_distance(a, b):
    return (((a["cx"] - b["cx"]) / XY_TOLERANCE_PX) ** 2
            + ((a["cy"] - b["cy"]) / XY_TOLERANCE_PX) ** 2
            + ((a["cz"] - b["cz"]) / Z_TOLERANCE_MM) ** 2)


def cluster_nodules(nodules):
    clusters = []
    for n in nodules:
        best, best_dist = None, None
        for c in clusters:
            if not ALLOW_SAME_READER_MERGE and n["reader"] in c["readers"]:
                continue
            if not is_close(n, c):
                continue
            d = scaled_distance(n, c)
            if best is None or d < best_dist:
                best, best_dist = c, d

        if best is None:
            best = {"members": [], "readers": set()}
            clusters.append(best)

        best["members"].append(n)
        best["readers"].add(n["reader"])
        k = len(best["members"])
        best["cx"] = sum(m["cx"] for m in best["members"]) / k
        best["cy"] = sum(m["cy"] for m in best["members"]) / k
        best["cz"] = sum(m["cz"] for m in best["members"]) / k
    return clusters


def make_row(base, sop, z, x_min, y_min, x_max, y_max, **extra):
    row = dict(base)
    row.update({
        "image_sop_id": sop,
        "z-slice": z,
        "width": x_max - x_min,
        "height": y_max - y_min,
        "x_centre": (x_min + x_max) / 2,
        "y_centre": (y_min + y_max) / 2,
    })
    row.update(extra)
    return row


def cluster_to_rows(cluster, merged_id):
    members = cluster["members"]
    first = members[0]
    base = {
        "patient_id": first["patient_id"],
        "series_instance_uid": first["series_instance_uid"],
        "merged_nodule_id": merged_id,
        "num_readers": len(cluster["readers"]),
        "malignancy": round(sum(m["malignancy"] for m in members) / len(members), 3),
    }

    rows = []
    if MERGE_SLICES:
        by_slice = {}
        for m in members:
            for s in m["slices"]:
                by_slice.setdefault((s["image_sop_id"], s["z"]), []).append(s)
        for (sop, z), boxes in by_slice.items():
            k = len(boxes)
            rows.append(make_row(
                base, sop, z,
                sum(b["x_min"] for b in boxes) / k,
                sum(b["y_min"] for b in boxes) / k,
                sum(b["x_max"] for b in boxes) / k,
                sum(b["y_max"] for b in boxes) / k,
                readers_on_slice=k,
            ))
    else:
        for m in members:
            for s in m["slices"]:
                rows.append(make_row(
                    base, s["image_sop_id"], s["z"],
                    s["x_min"], s["y_min"], s["x_max"], s["y_max"],
                    reader=m["reader"], original_nodule_id=m["nodule_id"],
                    reader_malignancy=m["malignancy"],
                ))
    return rows

if __name__ == "__main__":
    xml_folder_path = Path("/Volumes/Expansion/LIDC-XML/tcia-lidc-xml")
    project_folder = Path(__file__).parent.parent
    meta_data_path = project_folder / "data" / "metadata.csv"

    meta = load_meta_data(meta_data_path)
    xml_files = get_xml_files(xml_folder_path)

    nodules_by_series = {}
    for i, xml_path in enumerate(xml_files):
        for n in parse_xml(xml_path, meta):
            nodules_by_series.setdefault(n["series_instance_uid"], []).append(n)
        if i % 50 == 0:
            print(i)

    all_rows = []
    for uid, nodules in nodules_by_series.items():
        clusters = cluster_nodules(nodules)
        for merged_id, cluster in enumerate(clusters, start=1):
            all_rows.extend(cluster_to_rows(cluster, merged_id))

    df = pd.DataFrame(all_rows)
    df = df.sort_values(["patient_id", "series_instance_uid", "merged_nodule_id", "z-slice"])
    df.to_csv(OUTPUT_CSV, index=False)

    print(f"Finished: {len(nodules_by_series)} series, {len(df)} rows -> {OUTPUT_CSV}")
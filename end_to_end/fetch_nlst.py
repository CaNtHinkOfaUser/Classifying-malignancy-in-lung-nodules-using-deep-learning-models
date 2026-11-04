# Ishaan

"""Download NLST CT scans from The Cancer Imaging Archive (TCIA) to test the end-to-end system.

NLST (National Lung Screening Trial) is a different set of patients from LIDC-IDRI,
scanned for lung-cancer screening, so it shows how the system behaves on scans it
has never seen in any form. It has no radiologist nodule outlines like LIDC-IDRI's,
so it tests that the system runs and what it finds, not how accurate it is.

TCIA's public API (no account needed) lists NLST patients and serves each series as a .zip.
Only CT series are fetched; the SEG (segmentation) series next to them are skipped.

Usage:
    .venv/bin/python end_to_end/fetch_nlst.py --list 10            show 10 patients' CT series and sizes
    .venv/bin/python end_to_end/fetch_nlst.py --series <SeriesInstanceUID> [--out end_to_end/nlst]
Then:
    .venv/bin/python end_to_end/nodule_pipeline.py end_to_end/nlst/<SeriesInstanceUID>.zip
or upload the .zip on the website (app.py).

Data use: NLST images on TCIA are shared under TCIA's data usage policy; cite the
NLST and TCIA if results from them are reported.
"""

import argparse
import json
import urllib.request
from pathlib import Path

API = "https://services.cancerimagingarchive.net/nbia-api/services/v1"


def get_json(path):
    with urllib.request.urlopen(f"{API}/{path}", timeout=120) as r:
        return json.load(r)


def list_series(n):
    for p in get_json("getPatient?Collection=NLST")[:n]:
        for s in get_json(f"getSeries?Collection=NLST&PatientID={p['PatientId']}"):
            if s.get("Modality") == "CT":
                print(f"{p['PatientId']:>8}  {s.get('ImageCount', '?'):>4} slices  "
                      f"{int(s.get('FileSize', 0)) / 1e6:6.0f} MB  {s['SeriesInstanceUID']}  "
                      f"{s.get('SeriesDescription', '')}")


def download(series_uid, out):
    out.mkdir(parents=True, exist_ok=True)
    target = out / f"{series_uid}.zip"
    print(f"downloading {series_uid} -> {target}")
    with urllib.request.urlopen(f"{API}/getImage?SeriesInstanceUID={series_uid}", timeout=600) as r, \
            open(target, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
    print(f"done: {target.stat().st_size / 1e6:.0f} MB")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="List or download NLST CT series from TCIA.")
    p.add_argument("--list", type=int, metavar="N", help="list the CT series of the first N patients")
    p.add_argument("--series", nargs="+", help="SeriesInstanceUID(s) to download")
    p.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "nlst")
    args = p.parse_args()
    if args.list:
        list_series(args.list)
    for uid in args.series or []:
        download(uid, args.out)

# Ishaan

"""Local website for the end-to-end system: upload one CT scan, get every nodule back.

Run:    .venv/bin/python end_to_end/app.py            then open http://localhost:8000
        .venv/bin/python end_to_end/app.py --port 8080

Only Python's standard library is used for the server, so nothing extra has to be
installed. The page (web/index.html) uploads the scan's files one by one -- a
folder of DICOM files, several .dcm files, or one .zip -- then asks the server to
analyse them and polls for progress. Uploads are kept in a temporary folder per
job and deleted when the analysis finishes.

The server listens on localhost only: the scans never leave this computer.
"""

import argparse
import json
import shutil
import sys
import tempfile
import threading
import traceback
import uuid
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nodule_pipeline import Pipeline       # noqa: E402

WEB = Path(__file__).resolve().parent / "web"
MAX_FILE = 200 * 1024 * 1024                # one upload request: a .zip of a whole scan fits
JOBS = {}                                   # job id -> {"dir", "status", "progress", "message", "result", "error"}
LOCK = threading.Lock()
PIPELINE = None                              # loaded once, in main()
RUN_LOCK = threading.Lock()                  # one analysis at a time: both models share the GPU


def new_job():
    job = uuid.uuid4().hex[:12]
    JOBS[job] = {"dir": Path(tempfile.mkdtemp(prefix="nodule_scan_")), "status": "uploading",
                 "progress": 0.0, "message": "Waiting for files", "result": None, "error": None, "files": 0}
    return job


def run_job(job):
    j = JOBS[job]

    def progress(frac, msg):
        with LOCK:
            j["progress"], j["message"] = round(frac, 3), msg

    try:
        with RUN_LOCK:
            progress(0.01, "Queued")
            j["status"] = "running"
            for z in list(j["dir"].rglob("*.zip")):          # zips are unpacked next to themselves
                with zipfile.ZipFile(z) as zf:
                    zf.extractall(z.parent / (z.stem + "_unzipped"))
                z.unlink()
            j["result"] = PIPELINE.analyse(j["dir"], progress)
            j["status"] = "done"
    except Exception as e:                                    # shown to the user on the page
        traceback.print_exc()
        j["status"], j["error"] = "error", str(e)
    finally:
        shutil.rmtree(j["dir"], ignore_errors=True)


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, fmt, *args):                        # keep the terminal quiet for file uploads
        if "/api/upload" not in (args[0] if args else ""):
            super().log_message(fmt, *args)

    def do_GET(self):
        url = urlparse(self.path)
        q = parse_qs(url.query)
        if url.path in ("/", "/index.html"):
            return self._send(200, (WEB / "index.html").read_bytes(), "text/html; charset=utf-8")
        if url.path == "/api/status":
            j = JOBS.get(q.get("job", [""])[0])
            if j is None:
                return self._send(404, {"error": "unknown job"})
            with LOCK:
                return self._send(200, {k: j[k] for k in ("status", "progress", "message", "result", "error")})
        return self._send(404, {"error": "not found"})

    def do_POST(self):
        url = urlparse(self.path)
        q = parse_qs(url.query)
        if url.path == "/api/job":
            return self._send(200, {"job": new_job()})

        j = JOBS.get(q.get("job", [""])[0])
        if j is None:
            return self._send(404, {"error": "unknown job"})

        if url.path == "/api/upload":
            size = int(self.headers.get("Content-Length", 0))
            if size > MAX_FILE:
                return self._send(413, {"error": "file too large"})
            # keep only a safe file name; folder structure inside the scan is not needed
            name = Path(q.get("name", ["file"])[0]).name or "file"
            target = j["dir"] / f"{j['files']:05d}_{name}"
            target.write_bytes(self.rfile.read(size))
            j["files"] += 1
            return self._send(200, {"ok": True, "files": j["files"]})

        if url.path == "/api/analyze":
            if j["status"] != "uploading":
                return self._send(409, {"error": "already started"})
            j["status"] = "queued"
            threading.Thread(target=run_job, args=(q["job"][0],), daemon=True).start()
            return self._send(200, {"ok": True})
        return self._send(404, {"error": "not found"})


def main():
    global PIPELINE
    p = argparse.ArgumentParser(description="Website for the end-to-end nodule system.")
    p.add_argument("--port", type=int, default=8000)
    args = p.parse_args()
    print("loading YOLO11s and the 3D CNNs ...", flush=True)
    PIPELINE = Pipeline()
    print(f"ready on {PIPELINE.dev}: open http://localhost:{args.port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()

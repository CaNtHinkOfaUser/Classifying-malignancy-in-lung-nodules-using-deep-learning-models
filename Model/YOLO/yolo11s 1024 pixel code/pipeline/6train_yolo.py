# Jairus

"""Step 6. Train the YOLO nodule detector.

Run after:  5build_yolo_dataset.py
Run from:   this folder (Model/YOLO/<version>/pipeline); ct.py comes from the project's pipeline/
Reads:      /Volumes/Expansion/processed/yolo_lidc/   images, labels, val_monitor.txt
Writes:     /Volumes/Expansion/processed/yolo_runs/<run name>/
                weights/best.pt, weights/last.pt, results.csv, args.yaml, plots
Usage:      python 6train_yolo.py --smoke                    2 epochs on 5% of train: does it run, how long is an epoch
            python 6train_yolo.py                            main model, YOLO26s
            python 6train_yolo.py --model yolo11s.pt         the control, same settings
            python 6train_yolo.py --resume <run folder>      carry on after a crash or a closed laptop
            add --stop-after 11 to pause resumably before a 12-hour session limit
            On Colab (GPU):  use Model/yolo_colab.ipynb, which sets everything up
                             (data from pack_for_colab.py, runs saved to Google Drive)
Takes:      M2 MacBook (mps): run --smoke first, it prints the minutes per epoch.
            A Colab T4 is several times faster.

Decisions (fixed here so YOLO26s and its YOLO11s control are trained identically):
  * Only the model differs between the two runs. Everything else is in SETTINGS,
    and imgsz/epochs/batch go into the run name, so two runs can't be compared
    by accident when they were trained differently.
  * No colour augmentation. R, G and B are the slices below, at and above the
    nodule; shifting hue or saturation would blend those slices, and changing
    brightness would break the fixed lung window (the same HU is always the
    same grey, see ct.to_uint8_window).
  * Zoom range 0.25, not YOLO's 0.5. Zooming out by half would shrink a 9 px
    nodule to 4-5 px, smaller than anything in the real data.
  * Left-right flips on (the two lungs mirror each other); upside-down flips
    and rotation off (every scan is taken with the patient the same way up).
  * Mosaic stays on (YOLO's default): it shows 4 slices per image, so more
    nodules per batch. It is switched off for the last 10 epochs.
  * After every epoch YOLO checks val_monitor.txt (every positive val slice
    plus sampled negatives) and keeps the best epoch as best.pt. That is only
    a progress check: the real score is per nodule over every val slice, from
    7eval_yolo.py.
"""

import argparse
import time
import warnings
from pathlib import Path

import torch
from ultralytics import YOLO

from yolo_config import YOLO_DIR, YOLO_RUNS

# Apple's GPU (mps) has no deterministic version of a few operations; PyTorch
# warns on every training step. Runs on a Mac are therefore not bit-for-bit
# repeatable, which is fine for a detector but worth knowing.
warnings.filterwarnings("ignore", message=".*does not have a deterministic implementation.*")

SETTINGS = dict(
    single_cls=True,
    seed=42,
    deterministic=True,
    patience=20,            # stop if the val check hasn't improved for 20 epochs
    hsv_h=0.0, hsv_s=0.0, hsv_v=0.0,
    scale=0.25,
    fliplr=0.5, flipud=0.0, degrees=0.0,
    mosaic=1.0, close_mosaic=10,
    plots=True,
)


def pick_device():
    if torch.cuda.is_available():
        return "0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def write_data_yaml(root):
    """dataset.yaml pointing at `root`, so the dataset folder can be moved (e.g. to Colab)."""
    for need in ["images/train", "labels/train", "val_monitor.txt"]:
        if not (root / need).exists():
            raise SystemExit(f"{root / need} not found. Is the drive plugged in, "
                             "and has 5build_yolo_dataset.py been run?")
    path = root / "dataset_train.yaml"
    path.write_text(f"path: {root}\n"
                    "train: images/train\n"
                    "val: val_monitor.txt\n"
                    "names:\n  0: nodule\n")
    return path


def record_image_format(root, run_dir):
    """Copy image_format.txt ("png" or "jpeg 95", from pack_for_colab.py) into the
    run, so 7eval_yolo.py turns val/test into the same format before scoring."""
    src = root / "image_format.txt"
    fmt = src.read_text() if src.exists() else "png\n"
    (Path(run_dir) / "image_format.txt").write_text(fmt)
    print(f"training images: {fmt.strip()}")


class TimeBudgetReached(Exception):
    pass


def add_stop_after(model, hours):
    """Pause training at the end of an epoch if the next one wouldn't finish
    within `hours`. YOLO has already saved last.pt for that epoch by then, and
    stopping with an exception skips its end-of-training clean-up, which would
    strip the optimizer out of last.pt and make the run impossible to resume.

    YOLO's own `time` setting is not used: it shortens the whole learning-rate
    schedule to fit the time, so the run would no longer match the control.
    """
    if not hours:
        return
    t0 = time.time()
    last = {"t": t0}

    def check(trainer):
        now = time.time()
        epoch_secs, last["t"] = now - last["t"], now
        if trainer.stop:            # finished or early-stopped anyway: let YOLO wrap up normally
            return
        if now - t0 + epoch_secs > hours * 3600:
            raise TimeBudgetReached(
                f"\nPaused after epoch {trainer.epoch + 1}: another epoch wouldn't fit in "
                f"{hours} h. Resume with: python 6train_yolo.py --resume {trainer.save_dir}")

    model.add_callback("on_fit_epoch_end", check)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolo26s.pt", help="yolo26s.pt (main) or yolo11s.pt (control)")
    ap.add_argument("--imgsz", type=int, default=640, help="640, or 1024 to enlarge small nodules")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=None, help="default 16 on a GPU, 8 on a Mac")
    ap.add_argument("--device", default=None, help="0 (CUDA), mps or cpu; default: best available")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--stop-after", type=float, default=None,
                    help="pause (resumably) before this many hours are up, e.g. 11 before a 12 h session limit")
    ap.add_argument("--root", type=Path, default=YOLO_DIR, help="folder made by 5build_yolo_dataset.py")
    ap.add_argument("--smoke", action="store_true", help="2 epochs on 5%% of train, to time an epoch")
    ap.add_argument("--resume", type=Path, default=None, help="run folder to carry on from")
    args = ap.parse_args()

    if args.resume:
        # The dataset and the run folder may have moved since the run started
        # (e.g. a new Colab session unpacks the data afresh), so both are passed again.
        # Every training setting comes from the checkpoint, unchanged.
        data = write_data_yaml(args.root)
        record_image_format(args.root, args.resume)
        model = YOLO(args.resume / "weights" / "last.pt")
        add_stop_after(model, args.stop_after)
        extra = {"device": args.device} if args.device else {}
        try:
            model.train(resume=True, data=str(data), save_dir=str(args.resume), workers=args.workers, **extra)
        except TimeBudgetReached as e:
            print(e)
        raise SystemExit

    device = args.device or pick_device()
    batch = args.batch or (16 if device not in ("mps", "cpu") else 8)
    data = write_data_yaml(args.root)

    name = f"{Path(args.model).stem}_img{args.imgsz}_e{args.epochs}_b{batch}"
    extra = {}
    if args.smoke:
        name = "smoke_" + name
        extra = dict(epochs=2, fraction=0.05, plots=False)

    print(f"model {args.model}  device {device}  batch {batch}  imgsz {args.imgsz}")
    print(f"data  {data}\nruns  {YOLO_RUNS / name}")

    # time the training and validation halves of each epoch separately: --smoke
    # trains on 5% of the images but still validates on all of val_monitor.txt
    stamps = []
    model = YOLO(args.model)
    for event in ["on_train_epoch_start", "on_train_epoch_end", "on_fit_epoch_end"]:
        model.add_callback(event, lambda trainer, e=event: stamps.append((e, time.time())))
    model.add_callback("on_train_start", lambda trainer: record_image_format(args.root, trainer.save_dir))
    add_stop_after(model, args.stop_after)

    t0 = time.time()
    try:
        model.train(**{**SETTINGS,
                       "data": str(data), "imgsz": args.imgsz, "epochs": args.epochs,
                       "batch": batch, "device": device, "workers": args.workers,
                       "project": str(YOLO_RUNS), "name": name, "exist_ok": args.smoke,
                       **extra})
    except TimeBudgetReached as e:
        print(e)
        raise SystemExit
    minutes = (time.time() - t0) / 60

    run = Path(model.trainer.save_dir)
    print(f"\nFinished in {minutes:.0f} min -> {run}")
    if args.smoke and len(stamps) >= 6:
        # the second epoch, once everything is warmed up
        (_, start), (_, trained), (_, validated) = stamps[3:6]
        per_epoch = ((trained - start) / 0.05 + (validated - trained)) / 60
        print(f"one full epoch would take about {per_epoch:.0f} min on {device} "
              f"(~{per_epoch * args.epochs / 60:.0f} h for {args.epochs} epochs, "
              "less if early stopping kicks in)")
    print(f"next: python 7eval_yolo.py --weights {run / 'weights' / 'best.pt'} --split val")

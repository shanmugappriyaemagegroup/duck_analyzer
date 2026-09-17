# from ultralytics import YOLO

# def main():
#     # nano model - fast, fits comfortably on a 6GB GPU like the RTX 4050
#     model = YOLO("yolov8n.pt")

#     model.train(
#         data=r"C:\Users\EmageVision\Documents\autodistill\dataset\data.yaml",
#         epochs=100,
#         imgsz=640,
#         batch=8,       # lower to 4 if you hit CUDA out-of-memory
#         device=0,      # use the GPU (RTX 4050)
#     )

# if __name__ == "__main__":
#     main()


# from ultralytics import YOLO

# def main():
#     # YOLO11 nano - newer architecture, fewer parameters than YOLOv8n,
#     # better accuracy/speed tradeoff, and pairs well with model.track()
#     # for multi-object tracking later.
#     model = YOLO("yolo11m.pt")

#     model.train(
#         data=r"C:\Users\EmageVision\Documents\autodistill\dataset\data.yaml",
#         epochs=200,
#         patience=50,
#         imgsz=640,
#         batch=8,       # lower to 4 if you hit CUDA out-of-memory
#         device=0,      # RTX 4050
#     )

# if __name__ == "__main__":
#     main()



"""
train.py - Duck detector training as a plain, importable function.

Two ways this gets used:

1. Imported directly by a Celery task (background, non-blocking from the
   API's point of view) -- see tasks.py:
        from train import run_training
        run_training(training_id, data_path, results_path,
                      progress_callback=self.report_progress)

2. Run standalone from the command line for local testing:
        python train.py --data_path ... --results_path ...
   (this just builds the same args and calls run_training() itself)

Progress reporting: run_training() takes an optional `progress_callback`
function. If you don't pass one, progress is printed as JSON to stdout
(same shape as before) so it still works stand-alone. If you DO pass one
(as the Celery task does), that callback receives every progress event
instead -- nothing is printed.

    progress_callback(step: str, progress_percent: float | None,
                       message: str, **extra)
"""

import os
os.environ.setdefault("MINT_HEADLESS", "1")
os.environ.setdefault("MPLBACKEND", "Agg")

import sys
import json
import time
import argparse
import traceback

import psutil
import torch
from ultralytics import YOLO


def _default_callback(step, progress_percent=None, message="", **extra):
    """Falls back to printing JSON when no callback was supplied (e.g. when
    running this script standalone from the command line)."""
    payload = {"MINT_STEP": step, "message": message}
    if progress_percent is not None:
        payload["progress_percent"] = round(float(progress_percent), 1)
    payload.update(extra)
    print(json.dumps(payload), flush=True)


def _file_callback(progress_file):
    """Used when this script is launched as a subprocess (no Celery). Each
    progress event overwrites `progress_file` with the latest state, atomically
    (write to a temp file then os.replace) so a poller reading it never sees a
    half-written JSON blob. The FastAPI side just reads this file on each
    /training/status poll -- no IPC beyond the filesystem needed."""
    def callback(step, progress_percent=None, message="", **extra):
        payload = {"MINT_STEP": step, "message": message}
        if progress_percent is not None:
            payload["progress_percent"] = round(float(progress_percent), 1)
        payload.update(extra)
        tmp_path = progress_file + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump(payload, f)
        os.replace(tmp_path, progress_file)
    return callback


class PhaseTimer:
    def __init__(self, report):
        self.report = report
        self._start = {}
        self.durations = {}

    def start(self, phase):
        self._start[phase] = time.time()
        self.report("phase_start", message=f"Starting phase: {phase}", phase=phase)

    def end(self, phase):
        t0 = self._start.pop(phase, None)
        dur = round(time.time() - t0, 2) if t0 is not None else None
        self.durations[phase] = dur
        self.report("phase_end", message=f"Finished phase: {phase}",
                     phase=phase, duration_sec=dur)


class TrainingMetrics:
    def __init__(self, total_epochs, report):
        self.total_epochs = total_epochs
        self.report = report
        self.gpu_available = torch.cuda.is_available()

    def system_snapshot(self):
        snap = {
            "cpu_percent": psutil.cpu_percent(interval=None),
            "ram_percent": psutil.virtual_memory().percent,
        }
        if self.gpu_available:
            try:
                snap["gpu_mem_used_mb"] = round(
                    torch.cuda.memory_allocated() / (1024 ** 2), 1)
                snap["gpu_mem_total_mb"] = round(
                    torch.cuda.get_device_properties(0).total_memory
                    / (1024 ** 2), 1)
            except Exception:
                pass
        return snap

    def epoch_progress(self, epoch, metrics_dict=None):
        pct = round(100.0 * epoch / self.total_epochs, 1)
        payload = {"epoch": epoch, "total_epochs": self.total_epochs}
        payload.update(self.system_snapshot())
        if metrics_dict:
            payload["metrics"] = metrics_dict
        self.report("training_epoch", progress_percent=pct,
                     message=f"Epoch {epoch}/{self.total_epochs}", **payload)


def run_training(training_id, data_path, results_path,
                  backbone="yolo11n.pt", epochs=200, patience=50,
                  imgsz=640, batch_size=8, device="0", resume=False,
                  progress_callback=None):
    """
    The main training function. Plain Python in, plain Python out --
    no argparse, no subprocess required to call this.

    Returns a dict: {"success": bool, "best_weights": str | None,
                      "phase_durations": dict, "error": str | None}
    Raises nothing itself -- errors are caught and returned in the dict, so
    a Celery task can check `result["success"]` without a try/except.
    """
    report = progress_callback or _default_callback
    timer = PhaseTimer(report)

    report("started", progress_percent=0,
           message="Training run starting", training_id=training_id,
           config={"data_path": data_path, "results_path": results_path,
                   "backbone": backbone, "epochs": epochs,
                   "patience": patience, "imgsz": imgsz,
                   "batch_size": batch_size, "device": device,
                   "resume": resume})

    try:
        timer.start("data_loading")
        if not os.path.isfile(data_path):
            raise FileNotFoundError(f"data.yaml not found: {data_path}")
        os.makedirs(results_path, exist_ok=True)
        timer.end("data_loading")

        timer.start("model_init")
        model = YOLO(backbone)
        metrics = TrainingMetrics(total_epochs=epochs, report=report)
        timer.end("model_init")

        def on_epoch_end(trainer):
            epoch = trainer.epoch + 1
            m = {}
            if getattr(trainer, "metrics", None):
                for k, v in trainer.metrics.items():
                    try:
                        m[k] = float(v)
                    except (TypeError, ValueError):
                        pass
            metrics.epoch_progress(epoch, metrics_dict=m)

        model.add_callback("on_fit_epoch_end", on_epoch_end)

        timer.start("training")
        project_dir = os.path.dirname(results_path) or results_path
        run_name = os.path.basename(results_path) or "train"
        model.train(
            data=data_path, epochs=epochs, patience=patience,
            imgsz=imgsz, batch=batch_size, device=device,
            project=project_dir, name=run_name,
            exist_ok=True, resume=resume,
        )
        timer.end("training")

        timer.start("saving_model")
        best_weights = os.path.join(results_path, "weights", "best.pt")
        report("saving_model", progress_percent=99,
               message="Saving artifacts...", best_weights=best_weights)
        timer.end("saving_model")

        report("completed", progress_percent=100,
               message="Training completed successfully",
               best_weights=best_weights, phase_durations=timer.durations)

        return {"success": True, "best_weights": best_weights,
                "phase_durations": timer.durations, "error": None}

    except Exception as e:
        report("failed", message=f"Training failed: {e}",
               error=str(e), traceback=traceback.format_exc())
        return {"success": False, "best_weights": None,
                "phase_durations": timer.durations, "error": str(e)}


def _parse_args():
    p = argparse.ArgumentParser(description="Duck detector training (standalone runner)")
    p.add_argument("--data_path", required=True)
    p.add_argument("--results_path", required=True)
    p.add_argument("--backbone-names", dest="backbone", default="yolo11n.pt")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--patience", type=int, default=50)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--device", default="0")
    p.add_argument("--session_id", default=None)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--progress_file", default=None,
                    help="If set, progress is written here as JSON (overwritten "
                         "on every event) instead of printed to stdout. Used when "
                         "this script is launched via subprocess.Popen by "
                         "training_service.py.")
    return p.parse_args()


if __name__ == "__main__":
    # standalone CLI entrypoint -- calls the same function above. Runs either
    # as a plain local test (prints JSON to stdout) or as a subprocess launched
    # by training_service.py (writes JSON to --progress_file instead).
    args = _parse_args()
    callback = _file_callback(args.progress_file) if args.progress_file else None
    result = run_training(
        training_id=args.session_id, data_path=args.data_path,
        results_path=args.results_path, backbone=args.backbone,
        epochs=args.epochs, patience=args.patience, imgsz=args.imgsz,
        batch_size=args.batch_size, device=args.device, resume=args.resume,
        progress_callback=callback)
    sys.exit(0 if result["success"] else 1)

"""
prepare_dataset.py - Stage 1 of the duck pipeline, as a plain function.

Same pattern as train.py: run_dataset_prep() is imported directly by a
Celery task and called with .delay() for background execution, with
progress reported through a callback instead of printed JSON.

    from prepare_dataset import run_dataset_prep
    run_dataset_prep(session_id, video_path, output_dir,
                      progress_callback=self.report_progress)

Still extracts every frame, then draft-labels with a zero-shot model
(Grounding DINO via autodistill). Pipeline PAUSES after this -- no
training happens here. See the module docstring history in this repo for
the full stage breakdown (extract -> draft-label -> human review -> train).
"""

import os
os.environ.setdefault("MINT_HEADLESS", "1")
os.environ.setdefault("MPLBACKEND", "Agg")

import sys
import json
import time
import glob
import argparse
import traceback

import cv2


def _default_callback(step, progress_percent=None, message="", **extra):
    payload = {"MINT_STEP": step, "message": message}
    if progress_percent is not None:
        payload["progress_percent"] = round(float(progress_percent), 1)
    payload.update(extra)
    print(json.dumps(payload), flush=True)


def _file_callback(progress_file):
    """Same pattern as train.py's _file_callback -- atomic overwrite of a
    JSON file, so training_service.py can poll it from the FastAPI process."""
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


def _extract_all_frames(video_path, frames_dir, timer, report):
    timer.start("extracting_frames")
    os.makedirs(frames_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None

    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        idx += 1
        out_path = os.path.join(frames_dir, f"frame_{idx:06d}.jpg")
        cv2.imwrite(out_path, frame)
        if idx % 50 == 0:
            pct = (idx / total * 100) if total else None
            report("extracting_frames", progress_percent=pct,
                   message=f"Extracted {idx} frames" + (f"/{total}" if total else ""),
                   frames_so_far=idx)
    cap.release()

    if idx == 0:
        raise RuntimeError(f"No frames could be read from {video_path}")

    timer.end("extracting_frames")
    report("extracting_frames", progress_percent=100,
           message=f"Done -- {idx} frames extracted", frames_total=idx)
    return idx


def _autolabel_frames(frames_dir, dataset_dir, duck_prompt, other_prompt,
                       box_threshold, text_threshold, timer, report):
    timer.start("auto_labeling")
    report("auto_labeling", progress_percent=0,
           message="Loading zero-shot base model (Grounding DINO)...")

    from autodistill.detection import CaptionOntology
    from autodistill.detection.detection_base_model import NmsSetting
    from autodistill_grounding_dino import GroundingDINO

    ontology = CaptionOntology({duck_prompt: "duck", other_prompt: "other"})
    base_model = GroundingDINO(ontology=ontology, box_threshold=box_threshold,
                               text_threshold=text_threshold)

    report("auto_labeling", progress_percent=10,
           message="Running draft detection over all frames...")

    base_model.label(
        input_folder=frames_dir, extension=".jpg", output_folder=dataset_dir,
        record_confidence=True, nms_settings=NmsSetting.CLASS_SPECIFIC,
    )

    timer.end("auto_labeling")
    data_yaml = os.path.join(dataset_dir, "data.yaml")
    num_boxes, num_low_conf = _confidence_summary(dataset_dir, low_conf_threshold=0.5)
    report("auto_labeling", progress_percent=100,
           message="Draft labeling complete", data_yaml=data_yaml,
           total_boxes=num_boxes, low_confidence_boxes=num_low_conf)
    return data_yaml, num_boxes, num_low_conf


def _confidence_summary(dataset_dir, low_conf_threshold=0.5):
    total, low = 0, 0
    for split in ("train", "valid"):
        pattern = os.path.join(dataset_dir, split, "labels", "confidence-*.txt")
        for path in glob.glob(pattern):
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    total += 1
                    try:
                        if float(line) < low_conf_threshold:
                            low += 1
                    except ValueError:
                        pass
    return total, low


def run_dataset_prep(session_id, video_path, output_dir,
                      duck_prompt="duck", other_prompt="not a duck",
                      box_threshold=0.35, text_threshold=0.25,
                      progress_callback=None):
    """
    Plain function: extract every frame from `video_path`, draft-label them,
    and return once ready for human review. Does NOT train.

    Returns a dict: {"success": bool, "data_yaml": str | None,
                      "frames_total": int, "total_boxes": int,
                      "low_confidence_boxes": int, "error": str | None}
    """
    report = progress_callback or _default_callback
    timer = PhaseTimer(report)

    report("started", progress_percent=0,
           message="Dataset preparation starting", session_id=session_id,
           config={"video_path": video_path, "output_dir": output_dir,
                   "duck_prompt": duck_prompt, "other_prompt": other_prompt,
                   "box_threshold": box_threshold, "text_threshold": text_threshold})

    try:
        timer.start("loading_data")
        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")
        frames_dir = os.path.join(output_dir, "frames")
        os.makedirs(output_dir, exist_ok=True)
        timer.end("loading_data")

        num_frames = _extract_all_frames(video_path, frames_dir, timer, report)

        data_yaml, num_boxes, num_low_conf = _autolabel_frames(
            frames_dir, output_dir, duck_prompt, other_prompt,
            box_threshold, text_threshold, timer, report)

        report("awaiting_review", progress_percent=100,
               message="Draft dataset ready for human review",
               session_id=session_id, frames_total=num_frames,
               total_boxes=num_boxes, low_confidence_boxes=num_low_conf,
               data_yaml=data_yaml, phase_durations=timer.durations)

        return {"success": True, "data_yaml": data_yaml,
                "frames_total": num_frames, "total_boxes": num_boxes,
                "low_confidence_boxes": num_low_conf, "error": None}

    except Exception as e:
        report("failed", message=f"Dataset preparation failed: {e}",
               error=str(e), traceback=traceback.format_exc())
        return {"success": False, "data_yaml": None, "frames_total": 0,
                "total_boxes": 0, "low_confidence_boxes": 0, "error": str(e)}


def _parse_args():
    p = argparse.ArgumentParser(description="Extract frames + draft-label a duck video (standalone runner)")
    p.add_argument("--video_path", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--duck_prompt", default="duck")
    p.add_argument("--other_prompt", default="not a duck")
    p.add_argument("--box_threshold", type=float, default=0.35)
    p.add_argument("--text_threshold", type=float, default=0.25)
    p.add_argument("--session_id", default=None)
    p.add_argument("--progress_file", default=None,
                    help="If set, progress is written here as JSON instead of "
                         "printed to stdout (used by training_service.py).")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    callback = _file_callback(args.progress_file) if args.progress_file else None
    result = run_dataset_prep(
        session_id=args.session_id, video_path=args.video_path,
        output_dir=args.output_dir, duck_prompt=args.duck_prompt,
        other_prompt=args.other_prompt, box_threshold=args.box_threshold,
        text_threshold=args.text_threshold, progress_callback=callback)
    sys.exit(0 if result["success"] else 1)

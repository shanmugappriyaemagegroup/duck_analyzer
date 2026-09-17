"""
find_weak_frames.py
Scan a video with your model and export the frames where detection was WEAK,
so you can re-label and retrain on them to fix count-flicker.

A frame is flagged if EITHER:
  - duck count != EXPECTED_DUCK_COUNT   (missed / split / extra detections), OR
  - any detection confidence < LOW_CONF (shaky, flicker-prone detections)

For each flagged frame it saves:
  weak_frames/raw/frame_00042.jpg         <- clean frame, for RE-LABELING
  weak_frames/annotated/frame_00042.jpg   <- with boxes+conf, to SEE the problem
  weak_frames/report.csv                  <- frame, count, reason, confidences
"""

import os
import csv
import cv2

# ------------------ EDIT THESE ------------------
VIDEO_PATH = r"C:\Users\EmageVision\Documents\duck_classification\DuckVidNew\only_ducks_5_scattered.mp4"
MODEL_PATH = r"C:\Users\EmageVision\Documents\autodistill\runs\detect\train-4\weights\best.pt"
OUTPUT_DIR = r"C:\Users\EmageVision\Documents\duck_classification\weak_frames"
EXPECTED_DUCK_COUNT = 18
DUCK_CLASS_ID = 0
OTHER_CLASS_ID = 1
CONF = 0.25          # low detection threshold so we CATCH the shaky ones too
LOW_CONF = 0.60      # detections below this are considered "weak/uncertain"
DEVICE = 0
# ------------------------------------------------

GREEN = (0, 200, 0)
RED = (0, 0, 255)
ORANGE = (0, 165, 255)
WHITE = (255, 255, 255)


def draw_box(frame, xyxy, label, color):
    x1, y1, x2, y2 = map(int, xyxy)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
    scale = max(0.4, frame.shape[1] / 1600.0)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
    cv2.putText(frame, label, (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, scale, WHITE, 1, cv2.LINE_AA)


def main():
    from ultralytics import YOLO

    raw_dir = os.path.join(OUTPUT_DIR, "raw")
    ann_dir = os.path.join(OUTPUT_DIR, "annotated")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(ann_dir, exist_ok=True)

    model = YOLO(MODEL_PATH)

    report_rows = []
    flagged = 0
    frame_idx = 0

    for r in model.predict(source=VIDEO_PATH, conf=CONF,
                           device=DEVICE, stream=True, verbose=False):
        frame_idx += 1
        raw = r.orig_img.copy()
        annotated = r.orig_img.copy()

        duck_confs = []
        other_confs = []
        low_conf_hits = []

        if r.boxes is not None and len(r.boxes) > 0:
            xyxys = r.boxes.xyxy.cpu().numpy()
            clss = r.boxes.cls.cpu().numpy().astype(int)
            confs = r.boxes.conf.cpu().numpy()
            for xyxy, c, cf in zip(xyxys, clss, confs):
                cf = float(cf)
                if c == DUCK_CLASS_ID:
                    duck_confs.append(cf)
                    lbl = f"duck {cf:.2f}"
                elif c == OTHER_CLASS_ID:
                    other_confs.append(cf)
                    lbl = f"other {cf:.2f}"
                else:
                    lbl = f"{c} {cf:.2f}"
                # color: red if this detection is low-confidence (the weak ones)
                color = ORANGE if cf < LOW_CONF else GREEN
                if cf < LOW_CONF:
                    low_conf_hits.append(round(cf, 2))
                draw_box(annotated, xyxy, lbl, color)

        detected = len(duck_confs)
        count_wrong = (detected != EXPECTED_DUCK_COUNT)
        has_low_conf = len(low_conf_hits) > 0

        # ---- flag rule: count wrong OR any weak detection ----
        if count_wrong or has_low_conf:
            flagged += 1
            reasons = []
            if count_wrong:
                reasons.append(f"count={detected}(exp{EXPECTED_DUCK_COUNT})")
            if has_low_conf:
                reasons.append(f"lowconf={low_conf_hits}")
            reason_str = ";".join(reasons)

            # banner on the annotated copy
            cv2.putText(annotated, f"F{frame_idx} {reason_str}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, RED, 2, cv2.LINE_AA)

            name = f"frame_{frame_idx:05d}.jpg"
            cv2.imwrite(os.path.join(raw_dir, name), raw)
            cv2.imwrite(os.path.join(ann_dir, name), annotated)

            report_rows.append({
                "frame": frame_idx,
                "duck_count": detected,
                "expected": EXPECTED_DUCK_COUNT,
                "count_wrong": count_wrong,
                "num_low_conf": len(low_conf_hits),
                "low_conf_values": low_conf_hits,
                "min_duck_conf": round(min(duck_confs), 3) if duck_confs else None,
                "reason": reason_str,
            })

        if frame_idx % 50 == 0:
            print(f"scanned {frame_idx} frames, flagged {flagged} so far...")

    # write report
    with open(os.path.join(OUTPUT_DIR, "report.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "frame", "duck_count", "expected", "count_wrong",
            "num_low_conf", "low_conf_values", "min_duck_conf", "reason"])
        w.writeheader()
        w.writerows(report_rows)

    print(f"\nDone. Scanned {frame_idx} frames, flagged {flagged} weak frames.")
    print(f"Raw frames (for re-labeling): {raw_dir}")
    print(f"Annotated frames (to inspect): {ann_dir}")
    print(f"Report: {os.path.join(OUTPUT_DIR, 'report.csv')}")


if __name__ == "__main__":
    main()
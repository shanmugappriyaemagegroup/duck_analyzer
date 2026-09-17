"""
export_retrain_frames.py

Export 50 valuable frames for retraining.

Frames are selected WITHOUT assuming an expected duck count.

Selection strategy:
1. Find frames with no/weak duck detections.
2. Give higher scores to frames with fewer detections.
3. Give additional score to low-confidence detections.
4. Spread selected frames apart to avoid duplicates.
5. Export:
      retrain/
          images/
          labels/
          annotated/
          classes.txt
          report.csv
"""

import os
import csv
import cv2


# ============================================================
# SETTINGS
# ============================================================

VIDEO_PATH = r"C:\Users\EmageVision\Documents\duck_classification\DuckVidNew\only_ducks1_more_scattered.mp4"

MODEL_PATH = r"C:\Users\EmageVision\Documents\autodistill\runs\detect\train-4\weights\best.pt"

OUTPUT_DIR = r"C:\Users\EmageVision\Documents\duck_classification\retrain"

DUCK_CLASS_ID = 0
OTHER_CLASS_ID = 1

CLASS_NAMES = ["duck", "other_toys"]

# Detection threshold
CONF = 0.25

# Below this confidence = weak detection
LOW_CONF = 0.60

# Number of frames you want
TARGET_FRAMES = 50

# Minimum distance between selected frames
MIN_GAP = 3

DEVICE = 0


# ============================================================
# COLORS
# ============================================================

GREEN = (0, 200, 0)
ORANGE = (0, 165, 255)
RED = (0, 0, 255)
WHITE = (255, 255, 255)


# ============================================================
# DRAW BOX
# ============================================================

def draw_box(frame, xyxy, label, color):

    x1, y1, x2, y2 = map(int, xyxy)

    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        color,
        2,
        cv2.LINE_AA
    )

    scale = max(0.4, frame.shape[1] / 1600.0)

    (tw, th), _ = cv2.getTextSize(
        label,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        1
    )

    cv2.rectangle(
        frame,
        (x1, y1 - th - 6),
        (x1 + tw + 4, y1),
        color,
        -1
    )

    cv2.putText(
        frame,
        label,
        (x1 + 2, y1 - 4),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        WHITE,
        1,
        cv2.LINE_AA
    )


# ============================================================
# YOLO FORMAT
# ============================================================

def to_yolo_line(cls_id, xyxy, w, h):

    x1, y1, x2, y2 = xyxy

    cx = ((x1 + x2) / 2) / w
    cy = ((y1 + y2) / 2) / h

    bw = (x2 - x1) / w
    bh = (y2 - y1) / h

    return f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"


# ============================================================
# MAIN
# ============================================================

def main():

    from ultralytics import YOLO

    model = YOLO(MODEL_PATH)

    # ========================================================
    # PASS 1
    # SCORE EVERY FRAME
    # ========================================================

    print("Pass 1: Finding frames with weak/missing duck detections...")

    scored = []

    frame_idx = 0

    for r in model.predict(
        source=VIDEO_PATH,
        conf=CONF,
        device=DEVICE,
        stream=True,
        verbose=False
    ):

        frame_idx += 1

        duck_confs = []

        if r.boxes is not None and len(r.boxes) > 0:

            clss = r.boxes.cls.cpu().numpy().astype(int)
            confs = r.boxes.conf.cpu().numpy()

            for c, cf in zip(clss, confs):

                if c == DUCK_CLASS_ID:

                    duck_confs.append(float(cf))

        # ----------------------------------------------------
        # NUMBER OF DUCKS DETECTED
        # ----------------------------------------------------

        detected_ducks = len(duck_confs)

        # ----------------------------------------------------
        # LOW CONFIDENCE DUCKS
        # ----------------------------------------------------

        weak_ducks = sum(
            1 for cf in duck_confs
            if cf < LOW_CONF
        )

        # ----------------------------------------------------
        # MINIMUM CONFIDENCE
        # ----------------------------------------------------

        if duck_confs:
            min_conf = min(duck_confs)
        else:
            min_conf = 0.0

        # ====================================================
        # WEAKNESS SCORE
        # ====================================================

        # No detections = very suspicious
        if detected_ducks == 0:

            score = 100

        else:

            # Fewer ducks = higher score
            score = 20 / detected_ducks

            # Weak detections increase score
            score += weak_ducks * 5

            # Very low confidence increases score
            if min_conf < LOW_CONF:
                score += (LOW_CONF - min_conf) * 10

        scored.append(
            (
                score,
                frame_idx,
                detected_ducks,
                weak_ducks
            )
        )

        if frame_idx % 100 == 0:
            print(f"  scored {frame_idx} frames...")

    total_frames = frame_idx

    # ========================================================
    # SORT WORST FRAMES FIRST
    # ========================================================

    scored.sort(
        key=lambda x: -x[0]
    )

    # ========================================================
    # SELECT TOP 50
    # ========================================================

    picked = []

    picked_set = set()

    for score, fidx, duck_count, weak_count in scored:

        if len(picked) >= TARGET_FRAMES:
            break

        # Avoid consecutive / near duplicate frames
        if any(
            abs(fidx - p) < MIN_GAP
            for p in picked_set
        ):
            continue

        picked.append(fidx)
        picked_set.add(fidx)

    picked_set = set(picked)

    print()
    print("=" * 60)
    print(f"Total video frames : {total_frames}")
    print(f"Frames selected    : {len(picked)}")
    print("=" * 60)

    # ========================================================
    # CREATE OUTPUT DIRECTORIES
    # ========================================================

    img_dir = os.path.join(
        OUTPUT_DIR,
        "images"
    )

    lbl_dir = os.path.join(
        OUTPUT_DIR,
        "labels"
    )

    ann_dir = os.path.join(
        OUTPUT_DIR,
        "annotated"
    )

    for d in (
        img_dir,
        lbl_dir,
        ann_dir
    ):
        os.makedirs(
            d,
            exist_ok=True
        )

    # ========================================================
    # SAVE CLASSES
    # ========================================================

    with open(
        os.path.join(
            OUTPUT_DIR,
            "classes.txt"
        ),
        "w"
    ) as f:

        f.write(
            "\n".join(CLASS_NAMES)
            + "\n"
        )

    # ========================================================
    # PASS 2
    # EXPORT SELECTED FRAMES
    # ========================================================

    print()
    print("Pass 2: Exporting selected frames...")

    report_rows = []

    frame_idx = 0

    for r in model.predict(
        source=VIDEO_PATH,
        conf=CONF,
        device=DEVICE,
        stream=True,
        verbose=False
    ):

        frame_idx += 1

        if frame_idx not in picked_set:
            continue

        raw = r.orig_img.copy()

        annotated = r.orig_img.copy()

        h, w = raw.shape[:2]

        name = f"frame_{frame_idx:05d}"

        yolo_lines = []

        duck_count = 0

        weak_count = 0

        if r.boxes is not None and len(r.boxes) > 0:

            xyxys = r.boxes.xyxy.cpu().numpy()

            clss = r.boxes.cls.cpu().numpy().astype(int)

            confs = r.boxes.conf.cpu().numpy()

            for xyxy, c, cf in zip(
                xyxys,
                clss,
                confs
            ):

                c = int(c)

                cf = float(cf)

                # ------------------------------
                # YOLO label
                # ------------------------------

                yolo_lines.append(
                    to_yolo_line(
                        c,
                        xyxy,
                        w,
                        h
                    )
                )

                # ------------------------------
                # Duck count
                # ------------------------------

                if c == DUCK_CLASS_ID:

                    duck_count += 1

                    if cf < LOW_CONF:
                        weak_count += 1

                # ------------------------------
                # Draw box
                # ------------------------------

                if c < len(CLASS_NAMES):
                    class_name = CLASS_NAMES[c]
                else:
                    class_name = str(c)

                label = f"{class_name} {cf:.2f}"

                if cf < LOW_CONF:
                    color = ORANGE
                else:
                    color = GREEN

                draw_box(
                    annotated,
                    xyxy,
                    label,
                    color
                )

        # ====================================================
        # SAVE RAW IMAGE
        # ====================================================

        cv2.imwrite(
            os.path.join(
                img_dir,
                name + ".jpg"
            ),
            raw
        )

        # ====================================================
        # SAVE YOLO LABEL
        # ====================================================

        with open(
            os.path.join(
                lbl_dir,
                name + ".txt"
            ),
            "w"
        ) as f:

            f.write(
                "\n".join(yolo_lines)
                + ("\n" if yolo_lines else "")
            )

        # ====================================================
        # SAVE ANNOTATED PREVIEW
        # ====================================================

        cv2.putText(
            annotated,
            f"F{frame_idx} ducks={duck_count} weak={weak_count}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            RED,
            2,
            cv2.LINE_AA
        )

        cv2.imwrite(
            os.path.join(
                ann_dir,
                name + ".jpg"
            ),
            annotated
        )

        # ====================================================
        # REPORT
        # ====================================================

        report_rows.append(
            {
                "frame": frame_idx,
                "duck_count": duck_count,
                "weak_ducks": weak_count
            }
        )

    # ========================================================
    # SAVE REPORT
    # ========================================================

    with open(
        os.path.join(
            OUTPUT_DIR,
            "report.csv"
        ),
        "w",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "frame",
                "duck_count",
                "weak_ducks"
            ]
        )

        writer.writeheader()

        writer.writerows(
            report_rows
        )

    # ========================================================
    # DONE
    # ========================================================

    print()
    print("=" * 60)
    print("DONE")
    print("=" * 60)

    print(f"Exported frames : {len(report_rows)}")

    print(f"Raw images      : {img_dir}")

    print(f"YOLO labels     : {lbl_dir}")

    print(f"Annotated       : {ann_dir}")

    print(
        f"Report          : "
        f"{os.path.join(OUTPUT_DIR, 'report.csv')}"
    )

    print()
    print(
        "Load images + labels into CVAT / "
        "makesense.ai / X-AnyLabeling and correct the mistakes."
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()
import cv2
import json
import os

# ============================================================
# CONFIG
# ============================================================

VIDEO_PATH = r"C:\Users\EmageVision\Documents\duck_classification\edited_videos\cam_video2.avi"

ROI_JSON_PATH = r"C:\Users\EmageVision\Documents\autodistill\hand_roi.json"

VIDEO_WIDTH = 1920
VIDEO_HEIGHT = 1080

# ============================================================
# GLOBAL VARIABLES
# ============================================================

drawing = False
start_point = None
end_point = None
roi = None


# ============================================================
# MOUSE CALLBACK
# ============================================================

def draw_roi(event, x, y, flags, param):
    global drawing, start_point, end_point, roi

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        start_point = (x, y)
        end_point = (x, y)

    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        end_point = (x, y)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        end_point = (x, y)

        x1, y1 = start_point
        x2, y2 = end_point

        # Make coordinates positive regardless of drag direction
        x_min = min(x1, x2)
        y_min = min(y1, y2)
        x_max = max(x1, x2)
        y_max = max(y1, y2)

        w = x_max - x_min
        h = y_max - y_min

        if w > 0 and h > 0:
            roi = [x_min, y_min, w, h]


# ============================================================
# OPEN VIDEO
# ============================================================

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise RuntimeError(f"Could not open video:\n{VIDEO_PATH}")

# Force original resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, VIDEO_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, VIDEO_HEIGHT)

# Read first frame
ret, frame = cap.read()

if not ret:
    cap.release()
    raise RuntimeError("Could not read video.")

# Check actual resolution
actual_width = frame.shape[1]
actual_height = frame.shape[0]

print(f"Video resolution: {actual_width} x {actual_height}")

if actual_width != VIDEO_WIDTH or actual_height != VIDEO_HEIGHT:
    print(
        f"WARNING: Video is actually {actual_width}x{actual_height}, "
        f"not {VIDEO_WIDTH}x{VIDEO_HEIGHT}."
    )

# ============================================================
# WINDOW
# ============================================================

window_name = "Draw Hand ROI"

cv2.namedWindow(
    window_name,
    cv2.WINDOW_NORMAL
)

# Display at original size
cv2.resizeWindow(
    window_name,
    VIDEO_WIDTH,
    VIDEO_HEIGHT
)

cv2.setMouseCallback(
    window_name,
    draw_roi
)

paused = False

print("\n================================================")
print("ROI DRAWING")
print("================================================")
print("p  -> Pause / Resume video")
print("Draw ROI while paused")
print("r  -> Redraw ROI")
print("s  -> Save ROI")
print("q  -> Quit")
print("================================================\n")


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    if not paused:
        ret, frame = cap.read()

        if not ret:
            # Restart video when it reaches the end
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()

            if not ret:
                break

    display_frame = frame.copy()

    # --------------------------------------------------------
    # Draw ROI while selecting
    # --------------------------------------------------------

    if drawing and start_point and end_point:

        cv2.rectangle(
            display_frame,
            start_point,
            end_point,
            (0, 255, 0),
            3
        )

    # --------------------------------------------------------
    # Draw saved/current ROI
    # --------------------------------------------------------

    if roi is not None:

        x, y, w, h = roi

        cv2.rectangle(
            display_frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            3
        )

        cv2.putText(
            display_frame,
            f"ROI: x={x}, y={y}, w={w}, h={h}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2
        )

    # --------------------------------------------------------
    # Status
    # --------------------------------------------------------

    status = "PAUSED - Draw ROI" if paused else "PLAYING - Press P to pause"

    cv2.putText(
        display_frame,
        status,
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )

    # --------------------------------------------------------
    # Display
    # --------------------------------------------------------

    cv2.imshow(
        window_name,
        display_frame
    )

    key = cv2.waitKey(30 if not paused else 30) & 0xFF

    # --------------------------------------------------------
    # Pause / Resume
    # --------------------------------------------------------

    if key == ord("p"):

        paused = not paused

        if paused:
            print("Video PAUSED. Draw your ROI.")
        else:
            print("Video RESUMED.")

    # --------------------------------------------------------
    # Redraw
    # --------------------------------------------------------

    elif key == ord("r"):

        if paused:
            roi = None
            start_point = None
            end_point = None
            print("ROI cleared. Draw again.")
        else:
            print("Pause the video first using P.")

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    elif key == ord("s"):

        if not paused:
            print("Pause the video first using P.")
            continue

        if roi is None:
            print("No ROI selected. Draw an ROI first.")
            continue

        x, y, w, h = roi

        data = {
            "video_resolution": [
                VIDEO_WIDTH,
                VIDEO_HEIGHT
            ],
            "roi": [
                x,
                y,
                w,
                h
            ]
        }

        # Create directory if needed
        os.makedirs(
            os.path.dirname(ROI_JSON_PATH),
            exist_ok=True
        )

        with open(
            ROI_JSON_PATH,
            "w"
        ) as f:
            json.dump(
                data,
                f,
                indent=4
            )

        print("\nROI SAVED SUCCESSFULLY")
        print(f"Path: {ROI_JSON_PATH}")
        print(f"ROI: x={x}, y={y}, w={w}, h={h}")

        # Keep window open after saving

    # --------------------------------------------------------
    # Quit
    # --------------------------------------------------------

    elif key == ord("q") or key == 27:
        break


# ============================================================
# CLEANUP
# ============================================================

cap.release()
cv2.destroyAllWindows()

print("\nDone.")
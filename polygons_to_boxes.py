"""
polygons_to_boxes.py
Convert GroundedSAM's polygon/segmentation .txt labels
(class x1 y1 x2 y2 x3 y3 ...) into plain YOLO bounding boxes
(class cx cy w h), which every standard tool expects.
"""

import os

INPUT_DIR = r"C:\Users\EmageVision\Documents\autodistill\labels"
OUTPUT_DIR = r"C:\Users\EmageVision\Documents\autodistill\labels_bbox"

os.makedirs(OUTPUT_DIR, exist_ok=True)

converted = 0
for fname in os.listdir(INPUT_DIR):
    if not fname.lower().endswith(".txt"):
        continue

    in_path = os.path.join(INPUT_DIR, fname)
    out_path = os.path.join(OUTPUT_DIR, fname)

    out_lines = []
    with open(in_path, "r") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 5:
                continue

            cls = parts[0]
            coords = list(map(float, parts[1:]))

            # already a plain bbox line (class cx cy w h)
            if len(coords) == 4:
                out_lines.append(line.strip())
                continue

            # polygon: pairs of x,y -> take bounding extent
            xs = coords[0::2]
            ys = coords[1::2]
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)

            cx = (x_min + x_max) / 2
            cy = (y_min + y_max) / 2
            w = x_max - x_min
            h = y_max - y_min

            out_lines.append(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

    with open(out_path, "w") as f:
        f.write("\n".join(out_lines) + "\n")

    converted += 1

print(f"Converted {converted} files.")
print(f"New bbox labels saved to: {OUTPUT_DIR}")
import cv2
import os

video = r"C:\Users\EmageVision\Documents\duck_classification\edited_videos\cam_video3.avi"

out_dir = r"C:\Users\EmageVision\Documents\duck_classification\extracted frames"
every_n = 20

os.makedirs(out_dir, exist_ok=True)

# Enter the date manually
date = input("Enter date (YYYY-MM-DD): ")

cap = cv2.VideoCapture(video)

i = 0
saved = 0

while True:
    ok, frame = cap.read()

    if not ok:
        break

    if i % every_n == 0:
        filename = f"{date}_frame{saved:05d}.jpg"
        filepath = os.path.join(out_dir, filename)

        cv2.imwrite(filepath, frame)
        saved += 1

    i += 1

cap.release()

print(f"Saved {saved} frames to {out_dir}")
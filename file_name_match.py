import os
import re

# =========================
# FOLDERS
# =========================

anomaly_folder = r"C:\Users\EmageVision\Downloads\anomaly_frames_annotated_only_ducks_4_neatly_assembled (1)\anomaly_frames"

frames_folder = r"C:\Users\EmageVision\Downloads\anomaly_frames_annotated_only_ducks_4_neatly_assembled (1)\frames"


# =========================
# GET ANOMALY FRAME NUMBERS
# =========================

anomaly_numbers = set()

for filename in os.listdir(anomaly_folder):

    # Matches anomaly_frame_00012.jpg
    match = re.search(r"anomaly_frame_(\d+)", filename)

    if match:
        frame_number = match.group(1)
        anomaly_numbers.add(frame_number)


print(f"Found {len(anomaly_numbers)} anomaly frames.")


# =========================
# CHECK NORMAL FRAMES
# =========================

deleted = 0
kept = 0

for filename in os.listdir(frames_folder):

    # Matches frame_00012.jpg
    match = re.search(r"frame_(\d+)", filename)

    if not match:
        continue

    frame_number = match.group(1)

    file_path = os.path.join(frames_folder, filename)

    if frame_number in anomaly_numbers:
        # Keep matching frame
        kept += 1
        print(f"KEEP   : {filename}")

    else:
        # Delete non-matching frame
        os.remove(file_path)
        deleted += 1
        print(f"DELETE : {filename}")


# =========================
# SUMMARY
# =========================

print("\n=========================")
print("DONE")
print("=========================")
print(f"Kept   : {kept}")
print(f"Deleted: {deleted}")
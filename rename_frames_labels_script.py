import os

# ============================================================
# FOLDERS
# ============================================================

images_folder = r"C:\Users\EmageVision\Documents\labels\images_other_species"

labels_folder = r"C:\Users\EmageVision\Documents\labels\labels_other_species"


# ============================================================
# USER INPUT
# ============================================================

date = input("Enter date (YYYY-MM-DD): ").strip()

# Optional validation
try:
    from datetime import datetime
    datetime.strptime(date, "%Y-%m-%d")
except ValueError:
    print("Invalid date format!")
    print("Please enter date like: 2026-08-30")
    exit()


# ============================================================
# RENAME IMAGES
# ============================================================

image_count = 0

for filename in os.listdir(images_folder):

    old_path = os.path.join(images_folder, filename)

    # Skip folders
    if not os.path.isfile(old_path):
        continue

    # Only rename files starting with frame_
    if filename.startswith("frame_"):

        new_filename = f"{date}_{filename}"

        new_path = os.path.join(
            images_folder,
            new_filename
        )

        os.rename(old_path, new_path)

        print(f"IMAGE : {filename} -> {new_filename}")

        image_count += 1


# ============================================================
# RENAME LABELS
# ============================================================

label_count = 0

for filename in os.listdir(labels_folder):

    old_path = os.path.join(labels_folder, filename)

    # Skip folders
    if not os.path.isfile(old_path):
        continue

    # Only rename files starting with frame_
    if filename.startswith("frame_"):

        new_filename = f"{date}_{filename}"

        new_path = os.path.join(
            labels_folder,
            new_filename
        )

        os.rename(old_path, new_path)

        print(f"LABEL : {filename} -> {new_filename}")

        label_count += 1


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 50)
print("RENAME COMPLETE")
print("=" * 50)

print(f"Images renamed : {image_count}")
print(f"Labels renamed : {label_count}")
print(f"Date used      : {date}")
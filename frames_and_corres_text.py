import os

# =========================
# FOLDERS
# =========================

images_dir = r"C:\Users\EmageVision\Documents\labels\images_only1_more_scattered"
labels_dir = r"C:\Users\EmageVision\Documents\labels\labelsonly_1_ducks_more_scattered"

# Image extensions to check
image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}

deleted = 0
kept = 0

# =========================
# CHECK IMAGES
# =========================

for filename in os.listdir(images_dir):

    image_path = os.path.join(images_dir, filename)

    # Skip folders
    if not os.path.isfile(image_path):
        continue

    # Check if it is an image
    name, ext = os.path.splitext(filename)

    if ext.lower() not in image_extensions:
        continue

    # Corresponding label file
    label_path = os.path.join(labels_dir, name + ".txt")

    # If label does NOT exist → delete image
    if not os.path.exists(label_path):

        os.remove(image_path)

        print(f"Deleted image: {filename}")
        deleted += 1

    else:
        kept += 1

# =========================
# SUMMARY
# =========================

print("\n========== DONE ==========")
print(f"Images kept   : {kept}")
print(f"Images deleted: {deleted}")
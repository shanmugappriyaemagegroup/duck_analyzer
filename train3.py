from pathlib import Path
from ultralytics import YOLO

ROOT = Path(r"C:\Users\EmageVision\Documents\autodistill")
DATASET = ROOT / "dataset" / "dataset"          # folder with images/ and labels/
DATA_YAML = ROOT / "dataset" / "data.yaml"
OLD_WEIGHTS = ROOT / "runs" / "detect" / "train-12" / "weights" / "best.pt"

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}

YAML_TEXT = f"""path: {DATASET.as_posix()}
train: images/train
val: images/val
nc: 3
names:
  0: duck
  1: other_toys
  2: tray
"""


def write_yaml():
    DATA_YAML.write_text(YAML_TEXT, encoding="utf-8")
    print(f"Wrote {DATA_YAML}:\n{YAML_TEXT}")


def clear_cache():
    for cache in (DATASET / "labels").glob("*.cache"):
        cache.unlink()
        print(f"Deleted cache: {cache}")


def check_split(split):
    img_dir = DATASET / "images" / split
    lbl_dir = DATASET / "labels" / split
    if not img_dir.exists():
        raise FileNotFoundError(f"Image folder not found: {img_dir}")
    if not lbl_dir.exists():
        raise FileNotFoundError(f"Label folder not found: {lbl_dir}")

    imgs = [p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXT]
    missing = [p.name for p in imgs if not (lbl_dir / f"{p.stem}.txt").exists()]
    print(f"{split}: {len(imgs)} images, {len(missing)} missing labels")
    if missing:
        print("  Missing labels for:", missing[:10], "..." if len(missing) > 10 else "")
    return len(imgs), len(missing)


def main():
    if not OLD_WEIGHTS.exists():
        raise FileNotFoundError(f"Weights not found: {OLD_WEIGHTS}")

    write_yaml()
    clear_cache()

    n_train, miss_train = check_split("train")
    n_val, miss_val = check_split("val")
    print(f"Total: {n_train + n_val} images\n")

    if miss_train or miss_val:
        raise RuntimeError("Some images have no label file. Fix them before training.")

    model = YOLO(str(OLD_WEIGHTS))

    model.train(
        data=str(DATA_YAML),
        epochs=30,
        patience=10,
        imgsz=640,
        batch=8,
        device=0,
        freeze=10,
        optimizer="SGD",
        lr0=0.002,
        warmup_epochs=1,
        cache="ram",
        workers=4,
        project=str(ROOT / "runs" / "detect"),
        name="finetune_plus200",
        exist_ok=False,
    )

    new_weights = Path(model.trainer.best)
    print(f"\nNew best weights: {new_weights}")
    metrics = YOLO(str(new_weights)).val(data=str(DATA_YAML), imgsz=640, batch=8, device=0)
    print(f"mAP50:    {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")


if __name__ == "__main__":
    main()
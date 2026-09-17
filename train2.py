from ultralytics import YOLO

def main():
    # YOLO11 nano - newer architecture, fewer parameters than YOLOv8n,
    # better accuracy/speed tradeoff, and pairs well with model.track()
    # for multi-object tracking later.
    model = YOLO("yolo11m.pt")

    model.train(
        data=r"C:\Users\EmageVision\Documents\autodistill\dataset\data.yaml",
        epochs=200,
        patience=50,
        imgsz=640,
        batch=8,       # lower to 4 if you hit CUDA out-of-memory
        device=0,      # RTX 4050
    )

if __name__ == "__main__":
    main()
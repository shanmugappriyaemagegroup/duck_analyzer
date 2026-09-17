# auto_annotate.py
import argparse, os
from autodistill_grounded_sam import GroundedSAM
from autodistill.detection import CaptionOntology
from autodistill.utils import plot
import cv2

def build_ontology(pairs):
    d = {}
    for p in pairs:
        caption, cls = p.split("=", 1)
        d[caption.strip()] = cls.strip()
    return CaptionOntology(d)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", default="./dataset")
    ap.add_argument("--ext", default=".jpg")
    ap.add_argument("--ontology", nargs="+", required=True)
    ap.add_argument("--box-threshold", type=float, default=0.35)
    ap.add_argument("--text-threshold", type=float, default=0.25)
    ap.add_argument("--preview")
    args = ap.parse_args()

    ontology = build_ontology(args.ontology)
    base_model = GroundedSAM(
        ontology=ontology,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
    )

    if args.preview:
        detections = base_model.predict(args.preview)
        plot(
            image=cv2.imread(args.preview),
            classes=base_model.ontology.classes(),
            detections=detections,
        )
        print("Preview done.")
        return

    base_model.label(
        input_folder=args.input,
        output_folder=args.output,
        extension=args.ext,
    )
    print(f"Done. Dataset at: {os.path.abspath(args.output)}")

if __name__ == "__main__":
    main()
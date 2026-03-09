from ultralytics import YOLO
from pathlib import Path

MODEL_PATH = r"runs/detect/pothole_yolov82/weights/best.pt"
IMAGE_DIR = r"sample_images"

def main():
    model = YOLO(MODEL_PATH)
    out_dir = Path("inference_output")
    out_dir.mkdir(exist_ok=True)

    model.predict(
        source=IMAGE_DIR,
        imgsz=640,
        conf=0.25,
        save=True,
        project=str(out_dir),
        name="pothole_results"
    )

if __name__ == "__main__":
    main()

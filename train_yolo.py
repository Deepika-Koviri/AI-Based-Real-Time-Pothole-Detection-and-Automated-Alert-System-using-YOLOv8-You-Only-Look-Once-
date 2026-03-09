from ultralytics import YOLO

def main():
    model = YOLO("yolov8n.pt")  # or yolov8s.pt if GPU allows

    model.train(
        data=r"C:\Users\kovir\OneDrive\Desktop\FINAL YEAR PROJECT\pothole_reduced.yaml",
        imgsz=640,
        epochs=10,
        batch=16,
        name="pothole_yolov8",
        task="detect"
    )

if __name__ == "__main__":
    main()

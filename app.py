import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path

from midas_utils import get_depth_map_bgr
from pothole_metrics import estimate_volume, cost_time_rules

MODEL_PATH = r"runs/detect/pothole_yolov82/weights/best.pt"

@st.cache_resource
def load_models():
    yolo_model = YOLO(MODEL_PATH)
    return yolo_model

def main():
    st.title("AI‑Based Real‑Time Pothole Detection and Cost Estimation")

    uploaded = st.file_uploader("Upload road image", type=["jpg", "jpeg", "png"])
    if not uploaded:
        st.info("Upload a road image to start.")
        return

    file_bytes = np.frombuffer(uploaded.read(), np.uint8)
    bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    yolo = load_models()
    results = yolo.predict(source=bgr, imgsz=640, conf=0.25, verbose=False)[0]

    depth_map = get_depth_map_bgr(bgr)

    h, w = bgr.shape[:2]
    PIXEL_TO_CM = 0.5  # 1 pixel ≈ 0.5 cm on road (tune / calibrate) [file:23]
    pixel_area_cm2 = (PIXEL_TO_CM ** 2)

    pothole_rows = []
    annotated = bgr.copy()

    for i, box in enumerate(results.boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        conf = float(box.conf[0])

        # depth values inside the box
        depth_region = depth_map[y1:y2, x1:x2]
        area_pixels = depth_region.size
        area_cm2 = area_pixels * pixel_area_cm2

        volume_cm3 = estimate_volume(depth_region, area_cm2)
        cat, (c_min, c_max), minutes = cost_time_rules(volume_cm3)

        pothole_rows.append({
            "ID": i + 1,
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "Conf": conf,
            "Area_cm2": round(area_cm2, 2),
            "Volume_cm3": round(volume_cm3, 2),
            "Severity": cat,
            "Cost_Min": c_min,
            "Cost_Max": c_max,
            "Time_min": minutes,
        })

        # draw
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            annotated,
            f"{cat} {conf:.2f}",
            (x1, max(0, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

    st.subheader("Detections")
    st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), caption="Pothole detections with severity")

    if pothole_rows:
        import pandas as pd
        df = pd.DataFrame(pothole_rows)
        st.dataframe(df)
    else:
        st.warning("No potholes detected in this image.")

if __name__ == "__main__":
    main()

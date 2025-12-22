import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
import pandas as pd

from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter  # for polite usage of Nominatim [web:46]

from midas_utils import get_depth_map_bgr
from pothole_metrics import (
    estimate_volume_cm3,
    cost_time_from_volume,
)

MODEL_PATH = r"runs/detect/pothole_yolov82/weights/best.pt"


@st.cache_resource
def load_yolo():
    return YOLO(MODEL_PATH)


@st.cache_data(show_spinner=False)
def geocode_address(addr: str):
    """
    Use OpenStreetMap Nominatim to geocode a free‑text address.
    Returns (lat, lon) or (None, None) if not found.
    """
    if not addr.strip():
        return None, None

    geolocator = Nominatim(user_agent="pothole_app_geocoder")  # required ID [web:46][web:64]
    geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)
    location = geocode(addr)
    if location is None:
        return None, None
    return float(location.latitude), float(location.longitude)


def main():
    st.set_page_config(
        page_title="AI-Based Pothole Detection System",
        layout="wide",
    )

    # ----------------- HEADER -----------------
    st.markdown(
        """
        <h1 style="text-align:center;">🛣️ AI-Based Pothole Detection System</h1>
        <p style="text-align:center; font-size:16px;">
        Real-time Road Safety & Smart Monitoring
        </p>
        """,
        unsafe_allow_html=True,
    )

    # ----------------- LOCATION + ALERTS -----------------
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 📍 Location Services")
        st.write(
            "Enter the road location / address for the uploaded image "
            "(e.g., city, area, landmark)."
        )
        address = st.text_area(
            "Road address / location",
            value="",
            placeholder="Ex: Main Road, Near Bus Stand, Vijayawada, Andhra Pradesh",
        )
        st.session_state["current_address"] = address

    with col2:
        st.markdown("### 🔔 Smart Alerts")
        alerts_enabled = st.checkbox("Enable alerts", value=True)
        st.session_state["alerts_enabled"] = alerts_enabled

    st.markdown("---")

    # ----------------- LIVE MAP (BETWEEN LOCATION AND UPLOAD) -----------------
    st.markdown("### 🗺️ Live Map (OpenStreetMap)")
    st.write(
        "The typed address is geocoded using OpenStreetMap Nominatim and shown on this map. "
        "Detected potholes at this address are also plotted here."
    )

    # Geocode current address (if any)
    lat, lon = geocode_address(address) if address else (None, None)

    if "pothole_points" not in st.session_state:
        # store dicts: {"lat": ..., "lon": ..., "address": ...}
        st.session_state["pothole_points"] = []

    # Base list of points: current address marker (if geocoded) + stored potholes
    map_points = []
    if lat is not None and lon is not None:
        map_points.append({"lat": lat, "lon": lon})

    for p in st.session_state["pothole_points"]:
        map_points.append({"lat": p["lat"], "lon": p["lon"]})

    if map_points:
        df_map = pd.DataFrame(map_points)
        st.map(df_map, zoom=14)  # Streamlit uses Mapbox with OSM-like tiles [web:52][web:54]
    else:
        st.info("Enter an address to show it on the map.")

    st.markdown("---")

    # ----------------- SIDEBAR CONFIG -----------------
    st.sidebar.header("Cost configuration (India)")
    rate_min = st.sidebar.number_input(
        "Min rate (₹ per m³)", min_value=1000, max_value=20000,
        value=4000, step=500
    )
    rate_max = st.sidebar.number_input(
        "Max rate (₹ per m³)", min_value=1000, max_value=30000,
        value=6000, step=500
    )
    max_depth_cm = st.sidebar.slider(
        "Max pothole depth (cm)", min_value=5, max_value=30,
        value=10, step=1
    )
    pixel_to_cm = st.sidebar.number_input(
        "Approx. ground scale (cm per pixel)",
        min_value=0.1, max_value=5.0, value=0.5, step=0.1,
        help="Rough estimate: smaller value = camera closer to road.",
    )

    # ----------------- IMAGE UPLOAD -----------------
    st.subheader("Upload road image")
    uploaded = st.file_uploader(
        "Upload road image", type=["jpg", "jpeg", "png"]
    )
    if not uploaded:
        st.info("Upload a road image to start detection.")
        return

    file_bytes = np.frombuffer(uploaded.read(), np.uint8)
    bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # ----------------- YOLO + MIDAS INFERENCE -----------------
    yolo = load_yolo()
    results = yolo.predict(source=bgr, imgsz=640, conf=0.25, verbose=False)[0]
    depth_map = get_depth_map_bgr(bgr)

    pixel_area_cm2 = pixel_to_cm ** 2
    pothole_rows = []
    annotated = bgr.copy()
    h, w = bgr.shape[:2]

    for i, box in enumerate(results.boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        conf = float(box.conf[0])

        # Clamp to image bounds
        x1 = max(0, min(x1, w - 1))
        x2 = max(0, min(x2, w - 1))
        y1 = max(0, min(y1, h - 1))
        y2 = max(0, min(y2, h - 1))
        if x2 <= x1 or y2 <= y1:
            continue

        depth_region = depth_map[y1:y2, x1:x2]
        volume_cm3 = estimate_volume_cm3(
            depth_region,
            pixel_area_cm2=pixel_area_cm2,
            max_depth_cm=max_depth_cm,
        )

        severity, (cmin, cmax), minutes = cost_time_from_volume(
            volume_cm3,
            rate_min_rs_per_m3=rate_min,
            rate_max_rs_per_m3=rate_max,
        )

        area_cm2 = depth_region.size * pixel_area_cm2

        pothole_rows.append(
            {
                "ID": i + 1,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "Conf": round(conf, 3),
                "Area_cm2": round(area_cm2, 2),
                "Volume_cm3": round(volume_cm3, 2),
                "Severity": severity,
                "Cost_Min_₹": cmin,
                "Cost_Max_₹": cmax,
                "Time_min": minutes,
                "Address": address if address else "Not provided",
                "Lat": lat if lat is not None else None,
                "Lon": lon if lon is not None else None,
            }
        )

        # draw bbox
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            annotated,
            f"{severity} {conf:.2f}",
            (x1, max(0, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

    # ----------------- SHOW RESULTS -----------------
    st.subheader("Detections")
    st.image(
        cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
        caption="Pothole detections with severity",
    )

    if pothole_rows:
        df = pd.DataFrame(pothole_rows)
        st.caption(
            "Pothole detections with volume-based Indian cost estimation."
        )
        st.dataframe(df, use_container_width=True)

        # Add pothole points to map at geocoded location
        if lat is not None and lon is not None:
            st.session_state["pothole_points"].append(
                {"lat": lat, "lon": lon, "address": address}
            )

        # Smart alert
        if st.session_state.get("alerts_enabled", True):
            if any(row["Severity"] == "Large" for row in pothole_rows):
                st.warning("Alert: Large pothole detected at this location!")
    else:
        st.warning("No potholes detected in this image.")


if __name__ == "__main__":
    main()

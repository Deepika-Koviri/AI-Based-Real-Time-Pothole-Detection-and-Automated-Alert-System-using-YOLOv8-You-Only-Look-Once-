import os
from datetime import timedelta

from flask import (
    Flask, render_template, request, redirect,
    url_for, session, flash
)
from werkzeug.security import generate_password_hash, check_password_hash

from ultralytics import YOLO
import cv2
import numpy as np
import pandas as pd

from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter

from midas_utils import get_depth_map_bgr
from pothole_metrics import estimate_volume_cm3, cost_time_from_volume

import folium
from markupsafe import Markup

# ---------- CONFIG ----------
MODEL_PATH = r"runs/detect/pothole_yolov82/weights/best.pt"
UPLOAD_FOLDER = "static/uploads"
RESULT_FOLDER = "static/results"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

app = Flask(__name__)
app.secret_key = "change_this_secret_key"
app.permanent_session_lifetime = timedelta(hours=2)

# in-memory users
USERS = {}

# in-memory pothole points for map (append after each detection run)
POTHOLE_POINTS = []   # each item: {"lat":..., "lon":..., "severity":..., "label":...}

# ---------- HELPERS ----------
yolo_model = YOLO(MODEL_PATH)

geolocator = Nominatim(user_agent="pothole_app_geocoder")
geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)


def geocode_address(addr: str):
    if not addr.strip():
        return None, None
    location = geocode(addr)
    if location is None:
        return None, None
    return float(location.latitude), float(location.longitude)


# ---------- AUTH ROUTES ----------
@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        form_type = request.form.get("form_type")

        if form_type == "signup":
            name = request.form.get("name", "").strip()
            email = request.form.get("email", "").strip().lower()
            password = request.form.get("password", "")

            if not name or not email or not password:
                flash("All signup fields are required.", "danger")
                return redirect(url_for("home"))

            if email in USERS:
                flash("User already exists. Please login.", "warning")
                return redirect(url_for("home"))

            USERS[email] = {
                "name": name,
                "password": generate_password_hash(password)
            }
            flash("Signup successful. Please login now.", "success")
            return redirect(url_for("home"))

        elif form_type == "login":
            email = request.form.get("email_login", "").strip().lower()
            password = request.form.get("password_login", "")

            user = USERS.get(email)
            if not user or not check_password_hash(user["password"], password):
                flash("Invalid login credentials.", "danger")
                return redirect(url_for("home"))

            session.permanent = True
            session["user_email"] = email
            session["user_name"] = user["name"]
            flash("Logged in successfully.", "success")
            return redirect(url_for("dashboard"))

    return render_template("auth.html")


@app.route("/logout")
def logout():
    session.clear()
    flash("Logged out.", "info")
    return redirect(url_for("home"))


# ---------- MAIN DASHBOARD ----------
@app.route("/dashboard", methods=["GET", "POST"])
def dashboard():
    if "user_email" not in session:
        flash("Please login first.", "warning")
        return redirect(url_for("home"))

    detections_table = None
    annotated_rel_path = None

    if request.method == "POST":
        # no address textbox now
        address = ""
        alerts_enabled = bool(request.form.get("alerts_enabled"))

        rate_min = float(request.form.get("rate_min", 4000))
        rate_max = float(request.form.get("rate_max", 6000))
        max_depth_cm = float(request.form.get("max_depth_cm", 10))
        pixel_to_cm = float(request.form.get("pixel_to_cm", 0.5))

        file = request.files.get("image")
        if not file or file.filename == "":
            flash("Please upload a road image.", "danger")
            return redirect(url_for("dashboard"))

        filename = file.filename
        upload_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(upload_path)

        bgr = cv2.imread(upload_path)
        if bgr is None:
            flash("Error reading uploaded image.", "danger")
            return redirect(url_for("dashboard"))

        results = yolo_model.predict(source=bgr, imgsz=640, conf=0.25, verbose=False)[0]
        depth_map = get_depth_map_bgr(bgr)
        pixel_area_cm2 = pixel_to_cm ** 2

        h, w = bgr.shape[:2]
        annotated = bgr.copy()
        pothole_rows = []

        # If you later re-add address, you can geocode here:
        lat, lon = (None, None)

        for i, box in enumerate(results.boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0])

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

            pothole_rows.append({
                "ID": i + 1,
                "Conf": round(conf, 3),
                "Area_cm2": round(area_cm2, 2),
                "Volume_cm3": round(volume_cm3, 2),
                "Severity": severity,
                "Cost_Min": cmin,
                "Cost_Max": cmax,
                "Time_min": minutes,
            })

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

            # ---- OPTIONAL: append a demo map point (fake coords) ----
            # replace with real lat/lon from geopy or GPS later
            demo_lat = 17.6868 + 0.001 * i
            demo_lon = 83.2185 + 0.001 * i
            POTHOLE_POINTS.append({
                "lat": demo_lat,
                "lon": demo_lon,
                "severity": severity,
                "label": f"Pothole {i+1} - {severity}"
            })

        annotated_name = "result_" + filename
        annotated_path = os.path.join(RESULT_FOLDER, annotated_name)
        cv2.imwrite(annotated_path, annotated)
        annotated_rel_path = annotated_path

        if pothole_rows:
            detections_table = pothole_rows
            if alerts_enabled and any(r["Severity"] == "Large" for r in pothole_rows):
                flash("Alert: Large pothole detected!", "danger")
            elif alerts_enabled:
                flash("No large potholes detected in this image.", "info")
        else:
            flash("No potholes detected in this image.", "info")

    return render_template(
        "dashboard.html",
        user_name=session.get("user_name"),
        detections=detections_table,
        annotated_image=annotated_rel_path,
    )


# ---------- OPENSTREETMAP VIEW ----------
@app.route("/map")
def pothole_map():
    if "user_email" not in session:
        return redirect(url_for("home"))

    if not POTHOLE_POINTS:
        # fallback demo points if none collected yet
        sample_points = [
            {"lat": 17.6868, "lon": 83.2185, "severity": "Large",
             "label": "NH16 – major pothole"},
            {"lat": 17.7040, "lon": 83.3010, "severity": "Medium",
             "label": "City road – medium"},
            {"lat": 17.7150, "lon": 83.2100, "severity": "Small",
             "label": "Colony road – minor"},
        ]
    else:
        sample_points = POTHOLE_POINTS

    center_lat = sample_points[0]["lat"]
    center_lon = sample_points[0]["lon"]

    m = folium.Map(
        location=[center_lat, center_lon],
        tiles="OpenStreetMap",
        zoom_start=13,
    )

    def severity_color(sev: str) -> str:
        if sev == "Large":
            return "red"
        if sev == "Medium":
            return "orange"
        return "green"

    for p in sample_points:
        folium.Marker(
            location=[p["lat"], p["lon"]],
            popup=f"{p['label']} ({p['severity']})",
            icon=folium.Icon(color=severity_color(p["severity"]))
        ).add_to(m)

    map_html = m._repr_html_()  # HTML for embedding [web:34][web:36]
    return render_template("map.html", map_html=Markup(map_html))


if __name__ == "__main__":
    app.run(debug=True)

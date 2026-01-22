import os
import csv
import math
import base64
import requests
from pathlib import Path
from datetime import datetime
from urllib.parse import urlparse, urlunparse, parse_qs, urlencode

from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
from werkzeug.security import generate_password_hash, check_password_hash

import cv2
import numpy as np
from ultralytics import YOLO

from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter

from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError
from bson import ObjectId

import openrouteservice

from midas_utils import get_depth_map_bgr
from pothole_metrics import estimate_volume_cm3, cost_time_from_volume

from recommendation_model import PotholeRepairRecommender
import joblib

import pandas as pd
from pymongo import MongoClient
import joblib
from recommendation_model import PotholeRepairRecommender


import json
import os
from datetime import datetime

REPORTS_FILE = "reports.json"

def load_reports():
    if os.path.exists(REPORTS_FILE):
        try:
            with open(REPORTS_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return []
    return []

def save_reports(reports):
    with open(REPORTS_FILE, 'w', encoding='utf-8') as f:
        json.dump(reports, f, indent=2, default=str)



# ---------------- CONFIG ----------------
MODEL_PATH = r"runs/detect/pothole_yolov82/weights/best.pt"

UPLOAD_FOLDER = "static/uploads"
RESULT_FOLDER = "static/results"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

app = Flask(__name__)
app.secret_key = "change_this_secret_key"
app.permanent_session_lifetime = 60 * 60 * 2  # 2 hours


# ---------------- DB ----------------
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
mongo_client = MongoClient(MONGO_URI)
mongo_db = mongo_client["pothole_db"]

users_col = mongo_db["users"]
reports_col = mongo_db["reports"]
potholes_col = mongo_db["potholes"]

try:
    users_col.create_index("email", unique=True)
except Exception:
    pass


# ---------------- MODELS / SERVICES ----------------
yolo_model = YOLO(MODEL_PATH)

geolocator = Nominatim(user_agent="pothole_app_geocoder")
geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)

IMAGE_LOCATIONS_CSV = Path("image_locations_ap.csv")

ORS_API_KEY = os.getenv("ORS_API_KEY")
if ORS_API_KEY:
    ors_client = openrouteservice.Client(key=ORS_API_KEY)
    print("[STARTUP] ORS client initialized.")
else:
    ors_client = None
    print("[STARTUP] WARNING: ORS_API_KEY not set, routes will not be drawn via ORS.")


# ---------------- URL HELPERS ----------------
def safe_next(next_url: str):
    if not next_url:
        return url_for("dashboard")
    if next_url.startswith("http://") or next_url.startswith("https://"):
        return url_for("dashboard")
    if not next_url.startswith("/"):
        next_url = "/" + next_url
    return next_url


def add_modal(url: str, modal: str):
    p = urlparse(url)
    q = parse_qs(p.query)
    q["modal"] = [modal]
    new_query = urlencode(q, doseq=True)
    return urlunparse((p.scheme, p.netloc, p.path, p.params, new_query, p.fragment))


def remove_modal_param(url: str):
    p = urlparse(url)
    q = parse_qs(p.query)
    q.pop("modal", None)
    new_query = urlencode(q, doseq=True)
    return urlunparse((p.scheme, p.netloc, p.path, p.params, new_query, p.fragment))


# ---------------- GEO HELPERS ----------------
def geocode_once(text: str):
    if not text or not text.strip():
        return None, None
    try:
        loc = geocode(text, timeout=6)
    except Exception as e:
        print(f"[GEOCODE] error: {e}")
        return None, None
    if loc is None:
        return None, None
    return float(loc.latitude), float(loc.longitude)


def parse_location_from_form(form):
    location_text = (form.get("location_text") or "").strip()
    lat_str = (form.get("lat") or "").strip()
    lon_str = (form.get("lon") or "").strip()

    if lat_str and lon_str:
        try:
            return float(lat_str), float(lon_str), location_text or "User coordinates"
        except ValueError:
            return None, None, location_text

    if location_text:
        lat, lon = geocode_once(location_text)
        return lat, lon, location_text

    return None, None, ""


# ---------------- MAP / ROUTE HELPERS ----------------
def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = (math.sin(dphi / 2) ** 2 +
         math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2)
    return 2 * R * math.asin(math.sqrt(a))


def _project_xy_m(lat, lon, lat0):
    # Equirectangular approximation (OK for small areas like a city)
    k = 111320.0
    x = lon * math.cos(math.radians(lat0)) * k
    y = lat * k
    return x, y


def _point_to_segment_distance_m(px, py, ax, ay, bx, by):
    abx, aby = (bx - ax), (by - ay)
    apx, apy = (px - ax), (py - ay)
    ab2 = abx * abx + aby * aby
    if ab2 <= 1e-12:
        return math.hypot(px - ax, py - ay)

    t = (apx * abx + apy * aby) / ab2
    t = max(0.0, min(1.0, t))
    cx = ax + t * abx
    cy = ay + t * aby
    return math.hypot(px - cx, py - cy)


def min_distance_point_to_polyline(lat, lon, polyline_coords):
    """
    polyline_coords: list of [lon, lat]
    Returns minimum distance (meters) from point to any polyline segment.
    """
    if not polyline_coords or len(polyline_coords) < 2:
        return float("inf")

    lat0 = lat
    px, py = _project_xy_m(lat, lon, lat0)

    dmin = float("inf")
    for i in range(len(polyline_coords) - 1):
        lon1, lat1 = polyline_coords[i]
        lon2, lat2 = polyline_coords[i + 1]
        ax, ay = _project_xy_m(lat1, lon1, lat0)
        bx, by = _project_xy_m(lat2, lon2, lat0)
        d = _point_to_segment_distance_m(px, py, ax, ay, bx, by)
        if d < dmin:
            dmin = d
    return dmin


def get_route_geojson(start_lat, start_lon, end_lat, end_lon):
    if start_lat is None or start_lon is None or end_lat is None or end_lon is None:
        return None, None, "Start/Destination not found. Try adding 'Visakhapatnam' in both fields."

    # 1) Try ORS
    if ors_client is not None:
        try:
            coords = [[start_lon, start_lat], [end_lon, end_lat]]
            routes = ors_client.directions(
                coordinates=coords,
                profile="driving-car",
                format="geojson",
            )
            geom = routes["features"][0]["geometry"]
            return geom, "OpenRouteService", None
        except Exception as e:
            print(f"[ROUTE] ORS error: {e}")

    # 2) Fallback OSRM (no key)
    try:
        url = (
            "https://router.project-osrm.org/route/v1/driving/"
            f"{start_lon},{start_lat};{end_lon},{end_lat}"
            "?overview=full&geometries=geojson"
        )
        r = requests.get(url, timeout=12)
        r.raise_for_status()
        data = r.json()
        coords = data["routes"][0]["geometry"]["coordinates"]
        geom = {"type": "LineString", "coordinates": coords}
        return geom, "OSRM (fallback)", None
    except Exception as e:
        print(f"[ROUTE] OSRM error: {e}")
        return None, None, "Route service failed (ORS/OSRM). Check internet connection and try again."


# (kept, but not used on map now)
def load_csv_potholes():
    pts = []
    if not IMAGE_LOCATIONS_CSV.exists():
        return pts

    with IMAGE_LOCATIONS_CSV.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                lat = float(row["lat"])
                lon = float(row["lon"])
            except (KeyError, ValueError):
                continue
            pts.append({
                "lat": lat,
                "lon": lon,
                "place_name": row.get("place_name", "Pothole"),
                "severity": row.get("severity", "Small"),
                "source": "csv"
            })
    return pts


# ---------------- DETECTION ----------------
def process_image_detection(bgr, rate_min, rate_max, max_depth_cm, pixel_to_cm):
    results = yolo_model.predict(source=bgr, imgsz=640, conf=0.25, verbose=False)[0]
    depth_map = get_depth_map_bgr(bgr)

    pixel_area_cm2 = pixel_to_cm ** 2
    h, w = bgr.shape[:2]

    annotated = bgr.copy()
    rows = []

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

        rows.append({
            "ID": i + 1,
            "Conf": round(conf, 3),
            "Area_cm2": round(area_cm2, 2),
            "Volume_cm3": round(volume_cm3, 2),
            "Severity": severity,
            "Cost_Min": int(cmin),
            "Cost_Max": int(cmax),
            "Time_min": int(minutes),
            "x1": x1, "y1": y1, "x2": x2, "y2": y2
        })

        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            annotated,
            f"{severity} {conf:.2f}",
            (x1, max(0, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    return annotated, rows


def severity_rank(sev: str) -> int:
    if sev == "Large":
        return 3
    if sev == "Medium":
        return 2
    return 1


# ---------------- AUTO RECOMMENDATION HELPERS (NEW) ----------------
def build_auto_recommendations(limit=500):
    db_reports = list(reports_col.find().sort("created_at", -1).limit(limit))

    total_potholes = 0
    total_cost_min = 0
    total_cost_max = 0
    total_time_min = 0
    total_volume_cm3 = 0.0

    highest_sev = "Small"

    def sev_rank(sev):
        if sev == "Large":
            return 3
        if sev == "Medium":
            return 2
        return 1

    rows = []
    for r in db_reports:
        pothole_count = int(r.get("pothole_count", 0) or 0)
        cmin = int(r.get("total_cost_min", 0) or 0)
        cmax = int(r.get("total_cost_max", 0) or 0)
        tmin = int(r.get("total_time_min", 0) or 0)
        vol = float(r.get("total_volume_cm3", 0) or 0)

        total_potholes += pothole_count
        total_cost_min += cmin
        total_cost_max += cmax
        total_time_min += tmin
        total_volume_cm3 += vol

        sev = r.get("severity_max", "Small") or "Small"
        if sev_rank(sev) > sev_rank(highest_sev):
            highest_sev = sev

        created_at = r.get("created_at")
        if hasattr(created_at, "strftime"):
            created_at_str = created_at.strftime("%Y-%m-%d %H:%M:%S")
        else:
            created_at_str = str(created_at or "")

        rows.append({
            "report_id": str(r.get("_id")),
            "created_at": created_at_str,
            "place_name": r.get("location_text") or "Reported location",
            "lat": r.get("lat"),
            "lon": r.get("lon"),
            "severity": sev,
            "pothole_count": pothole_count,
            "total_volume_cm3": round(vol, 2),
            "total_cost_min": cmin,
            "total_cost_max": cmax,
            "total_time_min": tmin,
        })

    avg_volume_cm3 = (total_volume_cm3 / total_potholes) if total_potholes > 0 else 0.0
    cost_per_pothole_min = (total_cost_min / total_potholes) if total_potholes > 0 else 0.0
    cost_per_pothole_max = (total_cost_max / total_potholes) if total_potholes > 0 else 0.0
    time_per_pothole_min = (total_time_min / total_potholes) if total_potholes > 0 else 0.0

    summary = {
        "pothole_type": highest_sev,
        "total_potholes": total_potholes,
        "avg_volume_cm3": round(avg_volume_cm3, 2),
        "cost_per_pothole_min": int(round(cost_per_pothole_min, 0)),
        "cost_per_pothole_max": int(round(cost_per_pothole_max, 0)),
        "total_cost_min": total_cost_min,
        "total_cost_max": total_cost_max,
        "time_per_pothole_min": int(round(time_per_pothole_min, 0)),
        "total_time_min": total_time_min,
    }
    return summary, rows


# ---------------- AUTH ----------------
@app.route("/", methods=["GET"])
def home():
    return redirect(url_for("dashboard"))


@app.route("/auth", methods=["POST"])
def auth_action():
    form_type = request.form.get("form_type")
    next_url = safe_next(request.form.get("next") or request.referrer or url_for("dashboard"))

    if form_type == "signup":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")

        if not name or not email or not password:
            flash("All signup fields are required.", "danger")
            return redirect(add_modal(next_url, "signup"))

        try:
            users_col.insert_one({
                "name": name,
                "email": email,
                "password_hash": generate_password_hash(password),
                "created_at": datetime.utcnow()
            })
        except DuplicateKeyError:
            flash("User already exists. Please login.", "warning")
            return redirect(add_modal(next_url, "login"))

        flash("Signup successful. Please login now.", "success")
        return redirect(add_modal(next_url, "login"))

    if form_type == "login":
        email = request.form.get("email_login", "").strip().lower()
        password = request.form.get("password_login", "")

        user = users_col.find_one({"email": email})
        if not user or not check_password_hash(user.get("password_hash", ""), password):
            flash("Invalid login credentials.", "danger")
            return redirect(add_modal(next_url, "login"))

        session.permanent = True
        session["user_email"] = email
        session["user_name"] = user.get("name", "User")

        flash("Logged in successfully.", "success")
        clean_next = remove_modal_param(next_url)
        return redirect(clean_next)

    flash("Invalid request.", "danger")
    return redirect(url_for("dashboard"))


@app.route("/logout")
def logout():
    session.clear()
    flash("Logged out.", "info")
    return redirect(url_for("dashboard", modal="login"))


# @app.route('/recommendations')
# def recommendations():
#     user_name = session.get("user_name", "Guest")
    
#     # ✅ GLOBAL MONGODB QUERY - ALL users' reports (not user-specific)
#     from datetime import datetime
#     reports_cursor = db.reports.find({}).sort("created_at", -1).limit(50)
    
#     # ✅ Process for template with SEPARATE date/time columns
#     reported_table = []
#     for r in reports_cursor:
#         reported_table.append({
#             "image_url": r.get("image_url", "/static/pothole_sample.jpg"),
#             "date": r.get("created_at", datetime.now()).strftime("%d/%m/%Y"),  # ✅ DATE COLUMN
#             "time": r.get("created_at", datetime.now()).strftime("%H:%M"),      # ✅ TIME COLUMN  
#             "place_name": r.get("place_name", "Unknown"),
#             "lat": r.get("lat", 0),
#             "lon": r.get("lon", 0),
#             "severity": r.get("severity", "Medium"),
#             "pothole_count": r.get("pothole_count", 0),
#             "total_volume_cm3": r.get("total_volume_cm3", 0),
#             "total_cost_min": r.get("total_cost_min", 0),
#             "total_cost_max": r.get("total_cost_max", 0),
#             "total_time_min": r.get("total_time_min", 0),
#             "bboxes": r.get("bboxes", [])
#         })
    
#     # ✅ Keep your existing auto calculations (using global reports)
#     total_potholes = sum(r.get('pothole_count', 0) for r in reported_table)
#     total_volume = sum(r.get('total_volume_cm3', 0) for r in reported_table)
    
#     auto_data = {
#         'pothole_type': 'Medium',  # Add this for template
#         'total_potholes': total_potholes,
#         'avg_volume_cm3': total_volume / max(1, total_potholes) if total_potholes > 0 else 0,
#         'total_volume_cm3': total_volume,
#         'total_cost_min': f"₹{total_volume * 0.0003:.0f}",
#         'total_cost_max': f"₹{total_volume * 0.0005:.0f}",
#         'total_time_min': total_potholes * 15,  # 15 min per pothole estimate
#         'time_per_pothole_min': 15,
#         'cost_per_pothole_min': f"₹{total_volume * 0.0003 / max(1, total_potholes):.0f}",
#         'cost_per_pothole_max': f"₹{total_volume * 0.0005 / max(1, total_potholes):.0f}"
#     }
    
#     return render_template('recommendation.html',
#                          user_name=user_name,
#                          auto=auto_data,
#                          reported_table=reported_table)

from datetime import datetime


@app.route('/recommendations')
def recommendations():
    user_name = session.get("user_name", "Guest")
    
    # ✅ YOUR EXISTING GLOBAL DATA
    reports = globals().get('recommendation_reports', [])
    
    # ✅ FIXED: Handle string dates properly
    reported_table = []
    for r in reports[-50:][::-1]:  # Last 50 reports
        created_at_raw = r.get('created_at')
        
        # ✅ FIX: Convert string to datetime OR use now
        if isinstance(created_at_raw, str):
            try:
                created_at = datetime.fromisoformat(created_at_raw.replace('Z', '+00:00'))
            except:
                created_at = datetime.now()
        else:
            created_at = created_at_raw or datetime.now()
        
        reported_table.append({
            "image_url": r.get("image_url", "/static/pothole_sample.jpg"),
            "date": created_at.strftime("%d/%m/%Y"),           # ✅ 22/01/2026
            "time": created_at.strftime("%H:%M"),              # ✅ 17:54
            "place_name": r.get("place_name", "Unknown"),
            "lat": r.get("lat", 0),
            "lon": r.get("lon", 0),
            "severity": r.get("severity", "Medium"),
            "pothole_count": r.get("pothole_count", 0),
            "total_volume_cm3": r.get("total_volume_cm3", 0),
            "total_cost_min": r.get("total_cost_min", 0),
            "total_cost_max": r.get("total_cost_max", 0),
            "total_time_min": r.get("total_time_min", 0),
            "bboxes": r.get("bboxes", [])
        })
    
    # ✅ Your existing calculations
    total_potholes = sum(r.get('pothole_count', 0) for r in reported_table)
    total_volume = sum(r.get('total_volume_cm3', 0) for r in reported_table)
    
    auto_data = {
        'pothole_type': 'Medium',
        'total_potholes': total_potholes,
        'avg_volume_cm3': total_volume / max(1, total_potholes) if total_potholes > 0 else 0,
        'total_volume_cm3': total_volume,
        'total_cost_min': f"₹{total_volume * 0.0003:.0f}",
        'total_cost_max': f"₹{total_volume * 0.0005:.0f}",
        'total_time_min': total_potholes * 15,
        'time_per_pothole_min': 15,
        'cost_per_pothole_min': f"₹{total_volume * 0.0003 / max(1, total_potholes):.0f}",
        'cost_per_pothole_max': f"₹{total_volume * 0.0005 / max(1, total_potholes):.0f}"
    }
    
    return render_template('recommendation.html',
                         user_name=user_name,
                         auto=auto_data,
                         reported_table=reported_table)




import os
import uuid
from datetime import datetime
from flask import request, jsonify

@app.route('/upload_pothole', methods=['POST'])
def upload_pothole():
    if 'image' not in request.files:
        return jsonify({'success': False, 'error': 'No image'}), 400
    
    file = request.files['image']
    address = request.form.get('address', 'Unknown location')
    
    # Save image
    filename = f"pothole_{uuid.uuid4().hex[:8]}.jpg"
    filepath = f"static/uploads/{filename}"
    os.makedirs("static/uploads", exist_ok=True)
    file.save(filepath)
    
    # TODO: Add your YOLOv8 model here
    pothole_count = 2  # Replace with real detection
    volume_cm3 = 6500  # Replace with your MiDaS calculation
    
    # Add to our fallback data (simulates MongoDB)
    new_report = {
        'created_at': datetime.now().isoformat(),
        'place_name': address,
        'lat': float(request.form.get('lat', 17.6868)),
        'lon': float(request.form.get('lon', 83.1827)),
        'image_url': f'/static/uploads/{filename}',
        'bboxes': [[0.45, 0.35, 0.25, 0.30]],  # Replace with real YOLO bboxes
        'pothole_count': pothole_count,
        'total_volume_cm3': volume_cm3,
        'severity': 'Medium',
        'total_cost_min': volume_cm3 * 0.0003,
        'total_cost_max': volume_cm3 * 0.0005,
        'total_time_min': pothole_count * 20
    }
    
    # Append to global list (works without MongoDB)
    if 'pothole_reports' not in globals():
        globals()['pothole_reports'] = []
    globals()['pothole_reports'].append(new_report)
    
    return jsonify({
        'success': True, 
        'filename': filename, 
        'potholes': pothole_count,
        'address': address
    })



# ---------------- DASHBOARD ----------------
@app.route("/dashboard", methods=["GET", "POST"])
def dashboard():
    if "user_email" not in session:
        if request.method == "POST":
            flash("Please login first.", "warning")
            return redirect(url_for("dashboard", modal="login"))
        if request.args.get("modal") is None:
            return redirect(url_for("dashboard", modal="login"))
        return render_template("dashboard.html", user_name="Guest", detections=None, annotated_image=None)

    detections_table = None
    annotated_rel_path = None

    if request.method == "POST":
        # NEW: check which button was clicked
        action = request.form.get("action")

        alerts_enabled = bool(request.form.get("alerts_enabled"))
        rate_min = float(request.form.get("rate_min", 4000))
        rate_max = float(request.form.get("rate_max", 6000))
        max_depth_cm = float(request.form.get("max_depth_cm", 10))
        pixel_to_cm = float(request.form.get("pixel_to_cm", 0.5))

        lat, lon, location_text = parse_location_from_form(request.form)
        if lat is None or lon is None:
            flash("Please enter a valid address OR valid coordinates (lat/lon).", "warning")
            return redirect(url_for("dashboard"))

        input_type = request.form.get("input_type", "upload")
        bgr = None
        filename = None

        # === ORIGINAL WORKING CAMERA/UPLOAD CODE (no comments-only blocks) ===
        if input_type == "camera":
            camera_data = request.form.get("camera_image")
            if not camera_data:
                flash("Please capture an image from the camera.", "danger")
                return redirect(url_for("dashboard"))
            try:
                image_data = base64.b64decode(camera_data.split(",")[1])
                nparr = np.frombuffer(image_data, np.uint8)
                bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                filename = f"camera_{int(datetime.utcnow().timestamp())}.jpg"
            except Exception as e:
                flash(f"Error processing camera image: {str(e)}", "danger")
                return redirect(url_for("dashboard"))
        else:
            file = request.files.get("image")
            if not file or file.filename == "":
                flash("Please upload an image.", "danger")
                return redirect(url_for("dashboard"))

            base = os.path.basename(file.filename)
            filename = f"{int(datetime.utcnow().timestamp())}_{base}"
            upload_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(upload_path)

            bgr = cv2.imread(upload_path)
            if bgr is None:
                flash("Error reading image.", "danger")
                return redirect(url_for("dashboard"))

        annotated, pothole_rows = process_image_detection(
            bgr=bgr,
            rate_min=rate_min,
            rate_max=rate_max,
            max_depth_cm=max_depth_cm,
            pixel_to_cm=pixel_to_cm
        )

        annotated_name = "result_" + filename
        annotated_path = os.path.join(RESULT_FOLDER, annotated_name)
        cv2.imwrite(annotated_path, annotated)
        annotated_rel_path = annotated_path

        user_email = session["user_email"]
        created_at = datetime.utcnow()

        if pothole_rows:
            total_cost_min = sum(r["Cost_Min"] for r in pothole_rows)
            total_cost_max = sum(r["Cost_Max"] for r in pothole_rows)
            total_time_min = sum(r["Time_min"] for r in pothole_rows)
            total_volume = sum(r["Volume_cm3"] for r in pothole_rows)

            max_sev = "Small"
            for r in pothole_rows:
                if severity_rank(r["Severity"]) > severity_rank(max_sev):
                    max_sev = r["Severity"]

            report_doc = {
                "user_email": user_email,
                "image_file": filename,
                "result_file": annotated_name,
                "input_type": input_type,
                "location_text": location_text,
                "lat": float(lat),
                "lon": float(lon),
                "pothole_count": len(pothole_rows),
                "severity_max": max_sev,
                "total_cost_min": int(total_cost_min),
                "total_cost_max": int(total_cost_max),
                "total_time_min": int(total_time_min),
                "total_volume_cm3": float(total_volume),
                "created_at": created_at
            }
            report_id = reports_col.insert_one(report_doc).inserted_id

            # your existing potholes_col.insert_many(...) etc here

            detections_table = pothole_rows

            # NEW: auto-save to recommendation list when special button is used
            if action == "detect_and_save":
                recommendation_report = {
                    'created_at': created_at.isoformat(),
                    'user_name': session.get("user_name", user_email),
                    'user_email': user_email,
                    'place_name': location_text,
                    'lat': float(lat),
                    'lon': float(lon),
                    'image_url': annotated_rel_path,
                    'pothole_count': len(pothole_rows),
                    'total_volume_cm3': total_volume,
                    'severity': max_sev,
                    'total_cost_min': total_cost_min,
                    'total_cost_max': total_cost_max,
                    'total_time_min': total_time_min,
                    'report_id': str(report_id)
                }

                # if 'recommendation_reports' not in globals():
                #     globals()['recommendation_reports'] = []
                # globals()['recommendation_reports'].append(recommendation_report)

                # flash(f'✅ {len(pothole_rows)} potholes detected & saved to recommendations!', "success")

                # ✅ PERMANENT STORAGE
                reports = load_reports()
                reports.append(recommendation_report)
                save_reports(reports)
                globals()['recommendation_reports'] = reports  # Keep for compatibility

                flash(f'✅ {len(pothole_rows)} potholes detected & saved to recommendations!', "success")


            # your existing alerts_enabled / flash logic here

    return render_template(
        "dashboard.html",
        user_name=session.get("user_name"),
        detections=detections_table,
        annotated_image=annotated_rel_path
    )


# ---------------- RECOMMENDATION PAGE (UPDATED) ----------------

@app.route('/save_detection', methods=['POST'])
def save_detection():
    address = request.form.get('address', 'Vepagunta Road')
    potholes = int(request.form.get('potholes_detected', 0))
    
    # Create report using your EXISTING detection results
    new_report = {
        'created_at': datetime.now().isoformat(),
        'place_name': address,
        'lat': float(request.form.get('lat', 17.6868)),
        'lon': float(request.form.get('lon', 83.1827)),
        'image_url': request.form.get('image_url', '/static/pothole1.jpg'),
        'bboxes': [[0.45, 0.35, 0.25, 0.30]],  # From your real detection
        'pothole_count': potholes,
        'total_volume_cm3': potholes * 3500,  # From your MiDaS
        'severity': 'Medium',
        'total_cost_min': potholes * 1000,
        'total_cost_max': potholes * 1600,
        'total_time_min': potholes * 20
    }
    
    # Add to global list
    if 'pothole_reports' not in globals():
        globals()['pothole_reports'] = []
    globals()['pothole_reports'].append(new_report)
    
    return jsonify({
        'success': True,
        'potholes': potholes,
        'address': address
    })


# ---------------- MAP (ONLY REPORTED + ROUTE) ----------------
@app.route("/map", methods=["GET", "POST"])
def pothole_map():
    if "user_email" not in session:
        return redirect(url_for("dashboard", modal="login"))

    typed_start = ""
    typed_end = ""
    current_start = session.get("journey_start")
    current_end = session.get("journey_end")

    route_geojson = None
    route_source = None
    route_error = None

    if request.method == "POST":
        typed_start = (request.form.get("start_location") or "").strip()
        typed_end = (request.form.get("end_location") or "").strip()

        s_lat, s_lon = geocode_once(typed_start)
        e_lat, e_lon = geocode_once(typed_end)

        current_start = {"name": typed_start, "lat": s_lat, "lon": s_lon}
        current_end = {"name": typed_end, "lat": e_lat, "lon": e_lon}

        session["journey_start"] = current_start
        session["journey_end"] = current_end

        route_geojson, route_source, route_error = get_route_geojson(s_lat, s_lon, e_lat, e_lon)

    else:
        # Keep route visible on refresh if already stored in session
        if current_start and current_end:
            typed_start = current_start.get("name", "") or ""
            typed_end = current_end.get("name", "") or ""
            s_lat = current_start.get("lat")
            s_lon = current_start.get("lon")
            e_lat = current_end.get("lat")
            e_lon = current_end.get("lon")
            if s_lat is not None and s_lon is not None and e_lat is not None and e_lon is not None:
                route_geojson, route_source, route_error = get_route_geojson(s_lat, s_lon, e_lat, e_lon)

    # Load ONLY reported potholes from MongoDB
    db_reports = list(reports_col.find().sort("created_at", -1).limit(500))

    report_points = []
    for r in db_reports:
        lat = r.get("lat")
        lon = r.get("lon")
        if lat is None or lon is None:
            continue

        report_points.append({
            "lat": float(lat),
            "lon": float(lon),
            "place_name": r.get("location_text") or "Reported pothole",
            "severity": r.get("severity_max", "Small"),
            "source": "db",
            "report_id": str(r.get("_id")),
            "pothole_count": int(r.get("pothole_count", 0) or 0),
            "total_cost_min": int(r.get("total_cost_min", 0) or 0),
            "total_cost_max": int(r.get("total_cost_max", 0) or 0),
            "total_time_min": int(r.get("total_time_min", 0) or 0),
            "user_email": r.get("user_email", ""),
            "image_file": r.get("image_file", ""),
            "created_at": str(r.get("created_at", "")),
        })

    # Filter reported points near the route
    NEAR_THRESHOLD_M = 300  # was 150
    if route_geojson and route_geojson.get("type") == "LineString":
        coords_line = route_geojson["coordinates"]  # [lon,lat]
        potholes_on_route = []
        for p in report_points:
            dist_m = min_distance_point_to_polyline(p["lat"], p["lon"], coords_line)
            p["snap_dist_m"] = float(dist_m)
            if dist_m <= NEAR_THRESHOLD_M:
                potholes_on_route.append(p)
    else:
        potholes_on_route = report_points

    return render_template(
        "map.html",
        route_geojson=route_geojson,
        potholes=potholes_on_route,
        current_start=current_start,
        current_end=current_end,
        typed_start=typed_start,
        typed_end=typed_end,
        route_source=route_source,
        route_error=route_error,
    )


@app.route("/api/reports", methods=["GET"])
def api_reports():
    db_reports = list(reports_col.find().sort("created_at", -1).limit(500))
    out = []
    for r in db_reports:
        out.append({
            "id": str(r["_id"]),
            "lat": r.get("lat"),
            "lon": r.get("lon"),
            "place_name": r.get("location_text") or "Uploaded location",
            "severity": r.get("severity_max", "Small"),
            "pothole_count": r.get("pothole_count", 0),
            "total_cost_min": r.get("total_cost_min", 0),
            "total_cost_max": r.get("total_cost_max", 0),
        })
    return jsonify(out)


if __name__ == "__main__":
    app.run(debug=True)

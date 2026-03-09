import os
import csv
import math
import base64
import requests
from pathlib import Path
import time                    
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
from pywebpush import webpush, WebPushException

from flask_mail import Mail, Message
import os

app = Flask(__name__)
app.secret_key = "pothole_project_2026_super_secret_key_123"
app.permanent_session_lifetime = 60 * 60 * 2

from flask_mail import Mail, Message
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = 'kovirideepika@gmail.com'
app.config['MAIL_PASSWORD'] = 'bcro rwlj vbvk dkgr'
mail = Mail(app)

AUTHORITY_EMAILS = ['samhithameduri2005@gmail.com', 'navyadeepikasri@gmail.com']
print("✅ EMAIL CONFIG LOADED")

import requests
import json

ONESIGNAL_APP_ID = "fd315567-2a7f-46ef-a05e-b5c30469afc4"
ONESIGNAL_API_KEY = "os_v2_app_7uyvkzzkp5do7ic6wxbqi2npysy4gvxqmpce3h5njnav4z6b5ptc2gyjgd3bbm2i57iybvlqldqz2ootnui5hgxfoe553ykksy6eqyi"

def send_pothole_alert(count, location):
    headers = {"Content-Type": "application/json", "Authorization": f"Basic {ONESIGNAL_API_KEY}"}
    payload = {
        "app_id": ONESIGNAL_APP_ID,
        "included_segments": ["Subscribed Users"],
        "headings": {"en": "🚨 POTHOLE!"},
        "contents": {"en": f"{count} potholes at {location}"}
    }
    requests.post("https://onesignal.com/api/v1/notifications", headers=headers, json=payload)


VAPID_PUBLIC_KEY = 'BJyQkTwymLBN6lZRw4QK26njKEhbUDHBXJZT5w-6qREv-dP0zIoYlizjvxCc_ejrTRHfLuLXjO5tFWgIxS5rEU0'
VAPID_PRIVATE_KEY = 'MIGHAgEAMBMGByqGSM49AgEGCCqGSM49AwEHBG0wawIBAQQgazKELxS0ezJa3i-UH9Mj7qbcyv9FHOIEic7Vc9fJ6fehRANCAASckJE8MpiwTepWUcOECtup4yhIW1AxwVyWU-cPuqkRL_nT9MyKGJYs478QnP3o600R3y7i14zubRVoCMUuaxFN'

app.config['VAPID_PUBLIC_KEY'] = VAPID_PUBLIC_KEY
app.config['VAPID_PRIVATE_KEY'] = VAPID_PRIVATE_KEY


TWILIO_SID = "ACd1ef0c3b5e46d10327ed37e3ef4156df"
TWILIO_TOKEN = "faf489582eaa2b2fca34f2f0b68ad560" 
TWILIO_PHONE = "+17622144477"
AUTHORITY_PHONES = [
    "+919059847261"  
    # "+916301929516",  
    # "+919110564880" 
]
from twilio.rest import Client

import random

def send_sms_alert(detections, lat, lon):
    try:
        client = Client(TWILIO_SID, TWILIO_TOKEN)
        counts = {"Large": 0, "Medium": 0, "Small": 0}

        for d in detections:
            severity = d.get("Severity", "Unknown")
            counts[severity] = counts.get(severity, 0) + 1

        slogans = [
            "Small potholes cause big accidents.",
            "Better a moment of caution than a moment of regret.",
            "Safety isn't expensive. It's priceless.",
            "One second of caution can prevent hours of damage.",
            "Act fast, prevent accidents.",
            "Smooth roads, safe journeys.",
            "Report. Repair. Protect.",
            "Road safety starts with action.",
            "Don’t ignore the danger.",
            "Prevention is better than repair."
        ]

        selected_slogan = random.choice(slogans)

        message = f"""🚨 POTHOLE ALERT - VIZAG 
📍 {lat:.4f}, {lon:.4f}
Potholes: L:{counts['Large']} M:{counts['Medium']} S:{counts['Small']}
{selected_slogan}"""

        success_count = 0
        for phone in AUTHORITY_PHONES:
            sms = client.messages.create(
                body=message.strip(),
                from_=TWILIO_PHONE,
                to=phone
            )
            print(f"✅ SMS #{success_count+1} to {phone}! ID: {sms.sid}")
            success_count += 1
        
        print(f"🎉 {success_count}/3 SMS SENT!")
        return True
    except Exception as e:
        print(f"❌ SMS ERROR: {e}")
        return False




from flask import send_from_directory
app.static_folder = 'static'
@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory(app.static_folder, filename)
print("✅ STATIC FILES SERVING ENABLED")

UPLOAD_FOLDER = 'static/uploads'
RESULT_FOLDER = 'static/results'   

from pymongo import MongoClient
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
mongo_client = MongoClient(MONGO_URI)
mongodb = mongo_client.potholeapp  
reportscol = mongodb.reports       
userscol = mongodb.users
potholescol = mongodb.potholes

print("✅ MONGODB CONNECTED to potholeapp database!")
print(f"✅ Collections: reports({reportscol.count_documents({})}), users({userscol.count_documents({})})")


import base64
from datetime import datetime
from flask_mail import Mail, Message

import base64
from flask import url_for


# def send_pothole_alert(report):
#     import base64
#     import cv2
    
#     image_filename = report['image_url']  
#     image_path = f"static/uploads/{image_filename}"
    
#     print(f"🔍 Looking for NEW image: {image_path}")
    
#     image_html = '''
#     <div style="text-align:center;padding:20px;background:#ecfdf5;border-radius:12px;border:3px solid #22c55e;">
#         <p style="color:#15803d;font-weight:700;font-size:18px;">📸 NEW STREET IMAGE PROCESSED</p>
#     </div>
#     '''
    
#     if os.path.exists(image_path):
#         img = cv2.imread(image_path)
#         if img is not None:
#             img_small = cv2.resize(img, (200, 200), interpolation=cv2.INTER_AREA)
#             _, buffer = cv2.imencode('.jpg', img_small, [cv2.IMWRITE_JPEG_QUALITY, 50])
#             img_b64 = base64.b64encode(buffer).decode('utf-8')
            
#             image_html = f'''
#             <div style="text-align:center;padding:20px;background:#ecfdf5;border:3px solid #22c55e;border-radius:12px;">
#                 <img src="data:image/jpeg;base64,{img_b64}" 
#                      style="width:200px;height:200px;border-radius:10px;border:2px solid #facc15;">
#                 <p style="color:#15803d;font-weight:700;margin-top:8px;">📸 Fresh Street Photo</p>
#             </div>
#             '''
#             print("✅ NEW IMAGE SHOWS IN EMAIL!")
    
#     html_body = f"""
#     <!DOCTYPE html>
#     <html><head><meta charset="UTF-8"></head>
#     <body style="font-family:Segoe UI,sans-serif;margin:0;padding:20px;background:#f1f5f9;">
#         <div style="max-width:650px;margin:0 auto;background:#fff;border-radius:16px;box-shadow:0 10px 40px rgba(0,0,0,0.08);">
#             <div style="background:linear-gradient(135deg,#facc15,#fbbf24);color:white;padding:30px;text-align:center;">
#                 <h1 style="margin:0;font-size:28px;">🚨 PotholeAI Alert</h1>
#                 <p style="margin:5px 0 0;opacity:0.95;">Fresh Street Detection</p>
#             </div>
#             <div style="padding:35px;">
#                 <div style="display:grid;grid-template-columns:1fr 1fr;gap:20px;margin:25px 0;">
#                     <div style="background:#f8fafc;padding:25px;border-radius:12px;border-left:5px solid #facc15;">
#                         <h3 style="color:#1e293b;margin:0 0 15px;">📍 Location</h3>
#                         <p style="font-size:16px;">{report.get('place_name', 'N/A')}</p>
#                     </div>
#                     <div style="background:#f8fafc;padding:25px;border-radius:12px;border-left:5px solid #facc15;">
#                         <h3 style="color:#1e293b;margin:0 0 15px;">📏 Measurements</h3>
#                         <p><strong>Depth:</strong> <span style="color:#facc15;font-weight:700;">{report.get('depth',0)}m</span></p>
#                         <p><strong>Width:</strong> <span style="color:#facc15;font-weight:700;">{report.get('width',0)}m</span></p>
#                         <p><strong>Potholes:</strong> <span style="color:#ef4444;font-size:20px;">{report.get('pothole_count',0)}</span></p>
#                     </div>
#                 </div>
#                 <div style="text-align:center;padding:20px;background:linear-gradient(135deg,#10b981,#059669);color:white;border-radius:12px;font-size:18px;font-weight:600;">
#                     {report.get('message', 'Potholes detected')}
#                 </div>
#                 {image_html}
#             </div>
#             <div style="background:#f8fafc;padding:25px;text-align:center;color:#64748b;font-size:14px;">
#                 <p>{report.get('created_at', 'Now')} | PotholeAI</p>
#             </div>
#         </div>
#     </body></html>
#     """
    
#     msg = Message(
#         subject=f"🚨 PotholeAI: {report.get('pothole_count',0)} New Potholes Detected",
#         sender=app.config['MAIL_USERNAME'],
#         recipients=AUTHORITY_EMAILS,
#         html=html_body
#     )
    
#     mail.send(msg)
#     print("✅ FRESH STREET IMAGE SENT!")

def send_pothole_alert(report):
    import os
    from flask_mail import Message

    # 🔹 Fix image path properly
    image_url = report.get('image_url', '')

    if image_url.startswith("static/"):
        image_path = image_url
    else:
        image_path = os.path.join("static/uploads", image_url)

    print(f"📂 Looking for image at: {image_path}")

    # 🔹 HTML email body
    html_body = f"""
    <!DOCTYPE html>
    <html>
    <body style="font-family:Segoe UI,sans-serif;padding:20px;background:#f1f5f9;">
        <div style="max-width:650px;margin:auto;background:white;border-radius:16px;padding:30px;box-shadow:0 6px 20px rgba(0,0,0,0.1);">

            <h2 style="color:#ef4444;">🚨 PotholeAI Alert</h2>

            <p><strong>📍 Location:</strong> {report.get('place_name','N/A')}</p>
            <p><strong>📏 Depth:</strong> {report.get('depth',0)} m</p>
            <p><strong>📐 Width:</strong> {report.get('width',0)} m</p>
            <p><strong>🕳 Potholes:</strong> {report.get('pothole_count',0)}</p>

            <p style="margin-top:20px;font-weight:bold;">
                {report.get('message','Potholes detected')}
            </p>

            <h3 style="margin-top:25px;">📸 Detected Image</h3>

            <img src="cid:pothole_image"
                 style="width:100%;max-width:500px;border-radius:10px;border:1px solid #ddd;"/>

            <p style="margin-top:30px;font-size:14px;color:gray;">
                {report.get('created_at','Now')} | PotholeAI System
            </p>

        </div>
    </body>
    </html>
    """

    msg = Message(
        subject=f"🚨 PotholeAI: {report.get('pothole_count',0)} New Potholes Detected",
        sender=app.config['MAIL_USERNAME'],
        recipients=AUTHORITY_EMAILS
    )

    msg.html = html_body

    # 🔹 Attach image INLINE (not attachment)
    if os.path.exists(image_path):
        with open(image_path, 'rb') as img:
            msg.attach(
                filename="pothole.jpg",
                content_type="image/jpeg",
                data=img.read(),
                disposition="inline",
                # headers=[('Content-ID', '<pothole_image>')]
                headers={"Content-ID": "<pothole_image>"}
            )
        print("✅ Image embedded in email successfully!")
    else:
        print("⚠ Image not found. Email sent without image.")

    mail.send(msg)
    print("✅ Alert Email Sent Successfully!")


# def send_pothole_alert(report):
#     import os
#     from flask_mail import Message

#     image_url = report.get('image_url', '')
    
#     if image_url.startswith("static/"):
#         image_path = image_url
#     else:
#         image_path = os.path.join("static/uploads", image_url)

#     print(f"📂 Looking for image at: {image_path}")

#     html_body = f"""
#     <!DOCTYPE html>
#     <html>
#     <body style="font-family:Segoe UI,sans-serif;padding:20px;background:#f1f5f9;">
#         <div style="max-width:650px;margin:auto;background:white;border-radius:16px;padding:30px;">
#             <h2 style="color:#ef4444;">🚨 PotholeAI Alert</h2>

#             <p><strong>📍 Location:</strong> {report.get('place_name', 'N/A')}</p>
#             <p><strong>📏 Depth:</strong> {report.get('depth',0)} m</p>
#             <p><strong>📐 Width:</strong> {report.get('width',0)} m</p>
#             <p><strong>🕳 Potholes:</strong> {report.get('pothole_count',0)}</p>

#             <p style="margin-top:20px;font-weight:bold;">
#                 {report.get('message', 'Potholes detected')}
#             </p>

#             <p style="margin-top:30px;font-size:14px;color:gray;">
#                 {report.get('created_at', 'Now')} | PotholeAI System
#             </p>
#         </div>
#     </body>
#     </html>
#     """

#     msg = Message(
#         subject=f"🚨 PotholeAI: {report.get('pothole_count',0)} New Potholes Detected",
#         sender=app.config['MAIL_USERNAME'],
#         recipients=AUTHORITY_EMAILS
#     )

#     msg.html = html_body

#     if os.path.exists(image_path):
#         with open(image_path, 'rb') as img:
#             msg.attach(
#                 filename=os.path.basename(image_path),
#                 content_type='image/jpeg',
#                 data=img.read()
#             )
#         print("✅ Image attached successfully!")
#     else:
#         print("⚠ Image not found. Email sent without attachment.")

#     mail.send(msg)
#     print("✅ Alert Email Sent Successfully!")

from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify, abort
from functools import wraps

def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user_email' not in session:
            flash('Please login first.', 'warning')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated

def admin_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user_email' not in session:
            flash('Please login first.', 'warning')
            return redirect(url_for('login'))
        if not session.get('is_admin', False):
            flash('Admin access required.', 'error')
            return abort(403)
        return f(*args, **kwargs)
    return decorated

# 403 Error Handler (add right here too)
@app.errorhandler(403)
def forbidden(e):
    flash('Access denied. Admins only.', 'error')
    return redirect(url_for('index'))



@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')



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



MODEL_PATH = r"runs/detect/pothole_yolov82/weights/best.pt"

UPLOAD_FOLDER = "static/uploads"
RESULT_FOLDER = "static/results"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

app.secret_key = "change_this_secret_key"
app.permanent_session_lifetime = 60 * 60 * 2  # 2 hours


# ---------------- DB ----------------
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
mongo_client = MongoClient(MONGO_URI)
# mongo_db = mongo_client["pothole_db"]

# users_col = mongo_db["users"]
# reports_col = mongo_db["reports"]
# potholes_col = mongo_db["potholes"]
mongodb = mongo_client.potholeapp        # ← ONE DATABASE!

reports_col = mongodb.reports
users_col = mongodb.users
potholes_col = mongodb.potholes


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

def reverse_geocode(lat, lon):
    """Convert lat/lon → Human address (MVP Colony, Vizag)"""
    if not lat or not lon:
        return "Unknown location"
    
    try:
        # ✅ USE GLOBAL GEOLOCATOR (already imported!)
        location = geolocator.reverse(f"{lat},{lon}", timeout=5)
        if location and location.address:
            # ✅ SAFE SPLIT - NO CRASH!
            parts = [p.strip() for p in location.address.split(',') if p.strip()]
            return ', '.join(parts[:3])  # "MVP Colony, Visakhapatnam, Andhra Pradesh"
        return f"Lat:{lat:.4f}, Lon:{lon:.4f}"
    except Exception as e:
        print(f"❌ Reverse geocode failed: {e}")
        return f"Lat:{lat:.4f}, Lon:{lon:.4f}"



def parse_location_from_form(form):
    location_text = (form.get("location_text") or "").strip()
    lat_str = (form.get("lat") or "").strip()
    lon_str = (form.get("lon") or "").strip()
    
    # ✅ PRIORITY 1: Manual lat/lon → REVERSE GEOCODE!
    if lat_str and lon_str:
        try:
            lat, lon = float(lat_str), float(lon_str)
            # ✨ MAGIC: Convert coords → Address!
            readable_address = reverse_geocode(lat, lon)
            return lat, lon, readable_address  # "MVP Colony, Visakhapatnam"
        except ValueError:
            return None, None, location_text
    
    # Priority 2: Text address → Forward geocode (existing)
    if location_text:
        lat, lon = geocode_once(location_text)
        return lat, lon, location_text
    
    return None, None, ""


# def parse_location_from_form(form):
#     location_text = (form.get("location_text") or "").strip()
#     lat_str = (form.get("lat") or "").strip()
#     lon_str = (form.get("lon") or "").strip()

#     if lat_str and lon_str:
#         try:
#             return float(lat_str), float(lon_str), location_text or "User coordinates"
#         except ValueError:
#             return None, None, location_text

#     if location_text:
#         lat, lon = geocode_once(location_text)
#         return lat, lon, location_text

#     return None, None, ""


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

from werkzeug.security import generate_password_hash, check_password_hash

@app.route('/auth')
def auth():
    if 'user_email' in session:
        return redirect('/dashboard')
    return render_template('auth.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        email = request.form.get('email', '').strip()
        user_name = request.form.get('user_name', '').strip()
        password = request.form.get('password', '')
        
        print(f"🔍 FORM RECEIVED: email='{email}' | name='{user_name}' | pass='{password[:3]}...'")
        
        if not email or not password:
            flash('❌ Please fill all fields completely!', 'error')
            return render_template('auth.html')
        
        from pymongo import MongoClient
        from werkzeug.security import generate_password_hash
        from datetime import datetime
        
        print("🔗 Connecting to MongoDB...")
        client = MongoClient('mongodb://localhost:27017/')
        db = client['pothole_app']
        
        # Check duplicate
        if db.users.find_one({'email': email}):
            print(f"❌ Duplicate email: {email}")
            client.close()
            flash('❌ Email already registered!', 'error')
            return render_template('auth.html')
        
        print("💾 Saving user...")
        # SAVE USER
        result = db.users.insert_one({
            'email': email,
            'user_name': user_name,
            'password': generate_password_hash(password),
            'is_admin': False, 
            'created_at': datetime.utcnow().isoformat()
        })
        
        print(f"✅ SUCCESS! Saved {email} (ID: {str(result.inserted_id)[:24]}...)")
        client.close()
        flash('✅ Account created! Please sign in.', 'success')
        return redirect('/login')
    
    return render_template('auth.html')



@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        
        from pymongo import MongoClient
        client = MongoClient('mongodb://localhost:27017/')
        db = client['pothole_app']
        
        user = db.users.find_one({'email': email})
        client.close()
        
        # if user and check_password_hash(user['password'], password):
        #     session['user_email'] = email
        #     session['user_name'] = user['user_name']
        #     flash(f'✅ Welcome, {user["user_name"]}!', 'success')
        #     return redirect('/dashboard')
        if user and check_password_hash(user['password'], password):
            session['user_email'] = email
            session['user_name'] = user.get('user_name')  
            session['is_admin'] = user.get('is_admin', False)  
            flash(f"Welcome, {user.get('user_name')}", "success")
            return redirect(url_for('dashboard'))
        else:
            flash('❌ Invalid email or password!', 'error')
    
    return render_template('auth.html')

@app.route('/logout')
def logout():
    session.clear()
    flash('👋 Logged out successfully!', 'success')
    return redirect('/')


@app.route('/api/stats')
def get_stats():
    try:
        client = MongoClient('mongodb://localhost:27017/')
        db = client['pothole_app']
        reports = list(db.pothole_reports.find())  # ✅ Correct collection!
        total_potholes = sum(int(r.get('pothole_count', 0)) for r in reports)
        stats = {
            'potholes': total_potholes,
            'reports': len(reports),
            'accuracy': 89,
            'alerts_sent': sum(int(r.get('total_alerts', 0)) for r in reports)
        }
        print(f"📊 LIVE STATS: {total_potholes} potholes, {len(reports)} reports")
        client.close()
        return jsonify(stats)
    except:
        return jsonify({'potholes': 0, 'reports': 0, 'accuracy': 89, 'alerts_sent': 0})

@app.route('/api/alerts-stats')
def api_alerts_stats():
    try:
        client = MongoClient('mongodb://localhost:27017/')
        db = client['pothole_app']
        reports = list(db.pothole_reports.find().sort('last_updated', -1).limit(50))
        
        total_reports = len(reports)
        total_potholes = sum(int(r.get('pothole_count', 0)) for r in reports)
        total_emails = sum(int(r.get('total_alerts', 0)) for r in reports)
        total_sms = total_reports * 3  # 3 SMS per report
        
        # Severity breakdown
        severity_count = {'Large': 0, 'Medium': 0, 'Small': 0}
        for r in reports:
            sev = r.get('severity_max', 'Small')
            severity_count[sev] += 1
        
        stats = {
            'total_reports': total_reports,
            'total_potholes': total_potholes,
            'total_emails': total_emails,
            'total_sms': total_sms,
            'severity_breakdown': severity_count,
            'status_breakdown': {'sent': total_reports, 'repaired': 0}
        }
        print(f"📊 ALERTS STATS: {stats}")
        client.close()
        return jsonify(stats)
    except:
        return jsonify({
            'total_reports': 0, 'total_potholes': 0, 'total_emails': 0, 
            'total_sms': 0, 'severity_breakdown': {'Small': 2, 'Medium': 3, 'Large': 1}
        })

import re
from datetime import datetime

def generate_smart_pothole_id(place_name, reports_col):
    """🔥 Generate ID: Gajuwaka → GWK240301 (3 potholes = GWK240301)"""
    # Clean & extract location code
    clean_name = re.sub(r'[^a-zA-Z\s]', '', place_name.lower()).strip()
    words = clean_name.split()
    
    if len(words) >= 2:
        code = ''.join(word[0:3] for word in words[:2]).upper()[:6]  # GWK from Gajuwaka
    else:
        code = words[0][:6].upper() if words else 'UNK'
    
    # Date + Sequential number
    today = datetime.now().strftime('%d%m')  # 0903 (9th March)
    count = reports_col.count_documents({
        'new_pothole_id': {'$regex': f'^{code}{today}'}
    }) + 1
    
    return f"{code}{today}{count:03d}"  # GWK090301



@app.route('/alerts')
@admin_required
def alerts():
    from datetime import datetime
    from pymongo import MongoClient
    
    client = MongoClient('mongodb://localhost:27017/')
    db = client['pothole_app']  # ✅ CORRECT DATABASE
    reports_col = db.pothole_reports  # ✅ CORRECT COLLECTION

    reports = list(reports_col.find().sort('created_at', -1))
    print(f"🔍 Found {len(reports)} reports in pothole_reports collection")
    
    alerts_table = []
    for report in reports:
        # 🔥 GENERATE SMART ID IF MISSING
        if not report.get('new_pothole_id'):
            pothole_id = generate_smart_pothole_id(
                report.get('place_name', 'Unknown'), 
                reports_col
            )
            # Update database
            reports_col.update_one(
                {'_id': report['_id']},
                {'$set': {'new_pothole_id': pothole_id}}
            )
            print(f"✅ Generated ID: {pothole_id} for {report.get('place_name')}")
        else:
            pothole_id = report.get('new_pothole_id')
        
        created_at = report.get('created_at')
        if isinstance(created_at, str):  
            try:
                created_at = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
            except:
                created_at = datetime.now()
        elif created_at is None:
            created_at = datetime.now()
            
        alerts_table.append({
            'new_pothole_id': pothole_id,
            'report_id': str(report.get('_id')),
            'image_url': report.get('image_url', '/static/pothole_sample.jpg'),
            'date': created_at.strftime('%d/%m'),
            'time': created_at.strftime('%H:%M'),
            'place_name': report.get('place_name', 'Street Scan'),
            'lat': float(report.get('lat', 17.6868)),
            'lon': float(report.get('lon', 83.2185)),
            'severity': report.get('severity_max', 'Medium'),  # ✅ Use severity_max
            'pothole_count': int(report.get('pothole_count', 0)),
            'depth': float(report.get('depth', 0.12)),
            'width': float(report.get('width', 0.45)),
            'status': report.get('status', 'sent')
        })
    
    # Analytics
    total_reports = len(reports)
    total_potholes = sum(int(r.get('pothole_count', 0)) for r in reports)
    total_emails = sum(int(r.get('total_alerts', 0)) for r in reports)
    
    severity_breakdown = {'Small': 0, 'Medium': 0, 'Large': 0}
    for r in reports:
        sev = r.get('severity_max', 'Small')
        severity_breakdown[sev] = severity_breakdown.get(sev, 0) + 1
    
    analytics = {
        'total_reports': total_reports,
        'total_potholes': total_potholes,
        'total_emails': total_emails,
        'total_sms': total_reports * 3,  # 3 SMS per report
        'severity_breakdown': severity_breakdown
    }
    
    client.close()
    return render_template('alerts.html', alerts=alerts_table, analytics=analytics)


from pywebpush import webpush, WebPushException
import json
from pymongo import MongoClient
from datetime import datetime

def send_pothole_push_notification(title, body):
    """🚨 Send to ALL users in your users collection"""
    try:
        client = MongoClient('mongodb://localhost:27017/')
        db = client['pothole_app']
        
        all_users = list(db.users.find({}))
        
        sent_count = 0
        for user in all_users:
            if not user.get('email'):
                continue
                
            try:
                subscription = {
                    "endpoint": f"https://fcm.googleapis.com/fcm/send/{user['email']}",
                    "keys": {
                        "p256dh": "dummy_public_key_for_broadcast",
                        "auth": "dummy_auth_key_for_broadcast"
                    }
                }
                
                webpush(
                    subscription_info=subscription,
                    data=json.dumps({
                        'title': title,
                        'body': body,
                        'icon': '/static/icon.png',
                        'url': '/dashboard'
                    }),
                    vapid_private_key=app.config['VAPID_PRIVATE_KEY'],
                    vapid_claims={'sub': 'mailto:potholeapp@example.com'}
                )
                sent_count += 1
                print(f"🚨 Push sent to {user['email']}")
                
            except WebPushException:
                print(f"❌ Failed to send to {user['email']}")
        
        client.close()
        print(f"📢 Broadcast COMPLETE: {sent_count} users notified")
        return sent_count
        
    except Exception as e:
        print(f"❌ Broadcast error: {e}")
        return 0

@app.route('/test-broadcast')
def test_broadcast():
    """Test button - sends to ALL users"""
    count = send_pothole_push_notification(
        "🧪 TEST NOTIFICATION", 
        "This is a test broadcast to ALL users!"
    )
    return f"✅ Broadcast sent to {count} users! Check your browser notifications."

@app.route('/api/update_alert_status', methods=['POST'])
def update_alert_status():
    from flask import request, jsonify
    from bson import ObjectId
    
    data = request.get_json()
    report_id = data.get('report_id')
    status = data.get('status')
    
    if not report_id or not status:
        return jsonify({'success': False, 'error': 'Missing data'})
    
    try:
        from pymongo import MongoClient
        client = MongoClient('mongodb://localhost:27017/')
        db = client['pothole_app']
        
        if ObjectId.is_valid(report_id):
            result = db.pothole_reports.update_one(
                {'_id': ObjectId(report_id)},
                {'$set': {'status': status}}
            )
        else:
            result = db.pothole_reports.update_one(
                {'report_id': report_id},
                {'$set': {'status': status}}
            )
        
        client.close()
        return jsonify({'success': result.modified_count > 0})
    except Exception as e:
        print(f"Status update error: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/subscribe', methods=['POST'])
def subscribe():
    try:
        data = request.get_json()
        user_email = data['user_email']
        subscription = data['subscription']
        
        print(f"🔔 SERVER: Got subscription from {user_email}")
        
        from pymongo import MongoClient
        client = MongoClient('mongodb://localhost:27017/')
        db = client['pothole_app']
        
        db.users.update_one(
            {'email': user_email},
            {'$set': {'push_subscription': subscription}},
            upsert=True
        )
        client.close()
        
        print(f"✅ SAVED subscription for {user_email}")
        return jsonify({'success': True})
        
    except Exception as e:
        print(f"❌ Subscribe error: {e}")
        return jsonify({'error': str(e)}), 500

    
@app.route('/debug-subscriptions')
def debug_subs():
    client = MongoClient('mongodb://localhost:27017/')
    db = client['pothole_app']
    
    users = list(db.users.find({}, {'email':1, 'push_subscription':1}))
    
    html = "<h2>🔔 Push Subscriptions</h2><ul>"
    for user in users:
        sub_status = "✅ HAS SUB" if user.get('push_subscription') else "❌ NO SUB"
        html += f"<li>{user['email']}: {sub_status}</li>"
    html += "</ul>"
    
    client.close()
    return html

@app.route('/test-push')
def test_push():
    sent = send_pothole_push_notification("🧪 TEST", "Push notifications working!")
    return f"<h1>🚨 Sent to {sent} users!</h1>"


def send_pothole_push_notification(title, body):
    client = MongoClient('mongodb://localhost:27017/')
    db = client['pothole_app']
    
    users_with_subs = list(db.users.find({
        'push_subscription': {'$exists': True, '$ne': None}
    }, {'push_subscription': 1}))
    
    sent = 0
    for user in users_with_subs:
        try:
            webpush(
                subscription_info=user['push_subscription'],
                data=json.dumps({'title': title, 'body': body}),
                vapid_private_key=app.config['VAPID_PRIVATE_KEY'],
                vapid_claims={'sub': 'mailto:potholeai@example.com'}
            )
            sent += 1
            print(f"✅ PUSH SENT")
        except Exception as e:
            print(f"❌ Push failed: {e}")
    
    client.close()
    print(f"📢 Broadcast: {sent}/{len(users_with_subs)}")
    return sent


from datetime import datetime

from flask import session, render_template
from datetime import datetime
from pymongo import MongoClient  


@app.route('/blog', methods=['GET', 'POST'])
def blog():
    if request.method == 'POST':
        title = request.form.get('title')
        subtitle = request.form.get('subtitle', '')
        content = request.form.get('content', '')
        if title and content:
            mongodb.blogs.insert_one({
                'title': title, 'subtitle': subtitle, 'content': content,
                'created_at': datetime.utcnow().isoformat()
            })
            flash('✅ Blog published successfully! Refresh to see it listed.', 'success')
        else:
            flash('❌ Please fill title and content!', 'danger')
        return redirect(url_for('blog'))
    return render_template('blog.html')

import pickle

with open("pothole_recommender_v2.pkl", "rb") as f:
    recommender_data = pickle.load(f)

material_model = recommender_data["material_model"]
method_model = recommender_data["method_model"]
durability_model = recommender_data["durability_model"]

le_road = recommender_data["le_road"]
le_traffic = recommender_data["le_traffic"]
le_weather = recommender_data["le_weather"]
le_rain = recommender_data["le_rain"]
le_material = recommender_data["le_material"]
le_method = recommender_data["le_method"]

import pandas as pd

def predict_recommendation(road_type, depth, diameter, traffic, weather, rainfall):

    road_encoded = le_road.transform([road_type])[0]
    traffic_encoded = le_traffic.transform([traffic])[0]
    weather_encoded = le_weather.transform([weather])[0]
    rain_encoded = le_rain.transform([rainfall])[0]

    X = pd.DataFrame([{
        "Road_Type": road_encoded,
        "Pothole_Depth_cm": depth,
        "Pothole_Diameter_cm": diameter,
        "Traffic_Intensity": traffic_encoded,
        "Weather_Condition": weather_encoded,
        "Rainfall_Level": rain_encoded
    }])

    material_pred = material_model.predict(X)[0]
    method_pred = method_model.predict(X)[0]
    durability_pred = durability_model.predict(X)[0]

    material = le_material.inverse_transform([material_pred])[0]
    method = le_method.inverse_transform([method_pred])[0]

    return material, method, int(durability_pred)

@app.route('/recommendations')
@admin_required
def recommendations():
    user_name = session.get("user_name", "Guest")
    
    reports = []
    
    try:
        from pymongo import MongoClient
        client = MongoClient('mongodb://localhost:27017/')
        db = client['pothole_app']
        
        reports = list(db.pothole_reports.find().sort("created_at", -1))
        print(f"Found {len(reports)} MongoDB reports")  
        
        client.close()
    except Exception as e:
        print(f"MongoDB error: {e}")
    
    if not reports:
        reports = [{
            "image_url": "/static/pothole_sample.jpg",
            "created_at": "2026-01-23T12:00:00",
            "place_name": "MG Road, Hyderabad",
            "lat": 17.3850,
            "lon": 78.4867,
            "severity": "Medium",
            "pothole_count": 2,
            "total_volume_cm3": 12500,
            "total_cost_min": 3750,
            "total_cost_max": 6250,
            "total_time_min": 30,
            "bboxes": []
        }]
        print("Using demo data - no real reports found")
    
    reported_table = []
    for r in reports:
        created_at_raw = r.get('created_at')
        
        if isinstance(created_at_raw, str):
            try:
                created_at = datetime.fromisoformat(created_at_raw.replace('Z', '+00:00'))
            except:
                created_at = datetime.now()
        else:
            created_at = created_at_raw or datetime.now()
        
        reported_table.append({
            "image_url": r.get("image_url", "/static/pothole_sample.jpg"),
            "date": created_at.strftime("%d/%m/%Y"),
            "time": created_at.strftime("%H:%M"),
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


    road_type = "City Road"
    traffic = "Medium"
    weather = "Dry"
    rainfall = "Medium"

    
    if total_potholes > 0:
        avg_volume = total_volume / total_potholes
    else:
        avg_volume = 0

   

    import math

    diameter_cm = 40  

    if avg_volume > 0:
        depth_cm = avg_volume / (math.pi * (diameter_cm/2)**2)
    else:
        depth_cm = 5  

    print("ML INPUT → Depth:", depth_cm, "Diameter:", diameter_cm)

    try:
        material, method, durability = predict_recommendation(
            road_type,
            depth_cm,
            diameter_cm,
            traffic,
            weather,
            rainfall
        )
    except Exception as e:
        print("ML Prediction error:", e)
        material, method, durability = "Bitumen", "Manual Patching", 12

    auto_data["recommended_material"] = material
    auto_data["repair_method"] = method
    auto_data["expected_durability"] = durability
    
    return render_template('recommendation.html',
                           user_name=user_name,
                           auto=auto_data,
                           reported_table=reported_table)

def predict_recommendation(road_type, depth, diameter, traffic, weather, rainfall):
    import numpy as np

    road_encoded = le_road.transform([road_type])[0]
    traffic_encoded = le_traffic.transform([traffic])[0]
    weather_encoded = le_weather.transform([weather])[0]
    rain_encoded = le_rain.transform([rainfall])[0]

    X = np.array([[
        road_encoded,
        depth,
        diameter,
        traffic_encoded,
        weather_encoded,
        rain_encoded
    ]])

    material_pred = material_model.predict(X)[0]
    method_pred = method_model.predict(X)[0]
    durability_pred = durability_model.predict(X)[0]

    material = le_material.inverse_transform([material_pred])[0]
    method = le_method.inverse_transform([method_pred])[0]

    return material, method, int(durability_pred)


import re
from datetime import datetime

def generate_smart_pothole_id(place_name, reports_col):
    """🔥 Generate ID from ANY place_name → KMDY2402011"""
    clean_name = re.sub(r'[^a-zA-Z\s]', '', place_name.lower()).strip()
    words = clean_name.split()
    
    if len(words) >= 2:
        code = ''.join(word[0:3] for word in words[:2]).upper()
    else:
        code = words[0][:6].upper()
    
    today = datetime.now().strftime('%d%m')  # 2402
    count = reports_col.count_documents({
        'pothole_id': {'$regex': f'^{code}{today}'}
    }) + 1
    
    return f"{code}{today}{count:03d}"



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
    
    timestamp = int(datetime.utcnow().timestamp())
    base_name = secure_filename(file.filename)[:20]
    filename = f"{timestamp}_{base_name}.jpg"
    filepath = f"static/uploads/{filename}"
    os.makedirs("static/uploads", exist_ok=True)
    file.save(filepath)

    pothole_count = 2
    volume_cm3 = 6500
    
    from pymongo import MongoClient
    client = MongoClient('mongodb://localhost:27017/')
    db = client['pothole_app']
    reports_col = db.pothole_reports
    
    clean_name = re.sub(r'[^a-zA-Z\s]', '', address.lower()).strip()
    words = clean_name.split()
    code = ''.join(word[0:3] for word in words[:2]).upper() if len(words) >= 2 else words[0][:6].upper()
    today = datetime.now().strftime('%d%m')
    count = reports_col.count_documents({'pothole_id': {'$regex': f'^{code}{today}'}}) + 1
    pothole_id = f"{code}{today}{count:03d}"
    
    new_report = {
        'created_at': datetime.now().isoformat(),
        'new_pothole_id': pothole_id,  
        'place_name': address,
        'lat': float(request.form.get('lat', 17.6868)),
        'lon': float(request.form.get('lon', 83.1827)),
        'image_filename': filename,  
        'image_url': filename,
        'pothole_count': pothole_count,
        'depth': 0.12,  
        'width': 0.45,  
        'severity': 'Medium',
        'status': 'sent',  
        'total_volume_cm3': volume_cm3
    }
    
    reports_col.insert_one(new_report)  
    client.close()
    
    print(f"🆔 SAVED: {pothole_id} @ {address}")
    
    return jsonify({
        'success': True, 
        'filename': filename, 
        'potholes': pothole_count,
        'address': address,
        'pothole_id': pothole_id  
    })

# ---------------- DASHBOARD ----------------
@app.route("/dashboard", methods=["GET", "POST"])
@login_required
def dashboard():
    from datetime import datetime  
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
            filename = f"{int(time.time())}_{base}"
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
        created_at = time.time()

        if pothole_rows:
            total_cost_min = sum(r["Cost_Min"] for r in pothole_rows)
            total_cost_max = sum(r["Cost_Max"] for r in pothole_rows)
            total_time_min = sum(r["Time_min"] for r in pothole_rows)
            total_volume = sum(r["Volume_cm3"] for r in pothole_rows)

            max_sev = "Small"
            for r in pothole_rows:
                if severity_rank(r["Severity"]) > severity_rank(max_sev):
                    max_sev = r["Severity"]

            print(f"🔍 Found {len(pothole_rows)} potholes - SENDING SMS!")
            send_sms_alert(pothole_rows, lat, lon)
            print("✅ SMS SENT TO +919059847261!")

            report_doc = {
                "user_email": user_email,
                "image_file": filename,
                "original_filename": filename,
                "result_file": annotated_name,
                "input_type": input_type,
                "location_text": location_text,
                "lat": float(lat),
                "lon": float(lon),
                'original_filename': filename,
                "pothole_count": len(pothole_rows),
                "severity_max": max_sev,
                "total_cost_min": int(total_cost_min),
                "total_cost_max": int(total_cost_max),
                "total_time_min": int(total_time_min),
                "total_volume_cm3": float(total_volume),
                "total_alerts": 1,
                "created_at": created_at
    
            }
            report_id = reports_col.insert_one(report_doc).inserted_id
        


            if action == "detect_and_save":
                if pothole_rows:  
                    first_pothole = pothole_rows[0]
                    bbox_width_px = first_pothole['x2'] - first_pothole['x1']
                    pothole_width_m = round((bbox_width_px * pixel_to_cm) / 100, 2)
                    pothole_depth_m = round((first_pothole['Volume_cm3'] / first_pothole['Area_cm2']) / 1000, 2)
                    
                    total_cost_min = sum(r["Cost_Min"] for r in pothole_rows)
                    total_cost_max = sum(r["Cost_Max"] for r in pothole_rows)
                    total_time_min = sum(r["Time_min"] for r in pothole_rows)
                    total_volume = sum(r["Volume_cm3"] for r in pothole_rows)
                    
                    max_sev = "Small"
                    for r in pothole_rows:
                        if severity_rank(r["Severity"]) > severity_rank(max_sev):
                            max_sev = r["Severity"]
                    
                    recommendation_report = { 
                        'created_at': datetime.fromtimestamp(created_at).isoformat(),
                        'place_name': location_text,
                        'lat': float(lat),
                        'lon': float(lon),
                        'image_filename': filename,
                        'image_url': f"static/uploads/{filename}",
                        'pothole_count': len(pothole_rows),
                        'severity': max_sev,
                        'depth': max(0.05, pothole_depth_m),
                        'width': max(0.10, pothole_width_m),
                        'total_volume_cm3': total_volume,
                        'message': f"🚨 {len(pothole_rows)} potholes detected at {location_text}"
                    }
                    
                    flash(f'✅ {len(pothole_rows)} potholes found & alert sent!', "success")
                    
                else:  
                    recommendation_report = {
                        'created_at': datetime.fromtimestamp(created_at).isoformat(),
                        'place_name': location_text,
                        'lat': float(lat),
                        'lon': float(lon),
                        'image_url': f"static/uploads/{filename}",
                        'pothole_count': 0,
                        'severity': 'CLEAR',
                        'depth': 0,
                        'width': 0,
                        'total_volume_cm3': 0,
                        'message': '✅ Road Clear - No potholes detected!'
                    }
                    flash('✅ Road clear! Alert sent to authorities.', "success")
                
                    try:
                        send_pothole_alert(recommendation_report)
                        print("✅ EMAIL SENT!")
                    except Exception as e:
                        print(f"⚠️ Email failed (network): {e}")
                        print("🎉 UPLOAD STILL PERFECT - SMS worked!")
                    
                # Continue with rest of your code...
                reports = load_reports()
                # send_pothole_alert(recommendation_report)
                
                reports = load_reports()
                reports.append(recommendation_report)
                save_reports(reports)
                globals()['recommendation_reports'] = reports
                
                try:
                    from pymongo import MongoClient
                    from bson import ObjectId
                    import datetime
                    
                    client = MongoClient('mongodb://localhost:27017/')
                    db = client['pothole_app']
                    collection = db.pothole_reports
                    
                    report_data = recommendation_report.copy()
                    
                    existing = collection.find_one({"image_filename": report_data["image_filename"]})
                    
                    if existing:
                        result = collection.update_one(
                            {"image_filename": report_data["image_filename"]},
                            {
                                "$inc": {
                                    "pothole_count": report_data["pothole_count"],
                                    "total_alerts": 1,
                                    "total_volume_cm3": report_data.get("total_volume_cm3", 0)
                                },
                                "$set": {
                                    "last_updated": datetime.datetime.now(datetime.timezone.utc),
                                    "alert_sent": True
                                },
                                "$push": {
                                    "detection_history": {
                                        "timestamp": datetime.datetime.now(datetime.timezone.utc),
                                        "potholes": report_data["pothole_count"],
                                        "image": report_data["image_filename"]
                                    }
                                }
                            }
                        )
                        print(f"📈 INCREMENTED! +{report_data['pothole_count']} potholes (now {existing['pothole_count'] + report_data['pothole_count']})")
                    else:
                        collection.insert_one(report_data)
                        print("✅ NEW report saved!")
                    
                    client.close()
                except Exception as e:
                    print(f"❌ MongoDB error: {e}")

                
                
                reports = load_reports()
                reports.append(recommendation_report)
                save_reports(reports)
                globals()['recommendation_reports'] = reports
                
                try:
                    from pymongo import MongoClient
                    from bson import ObjectId
                    import datetime
                    
                    client = MongoClient('mongodb://localhost:27017/')
                    db = client['pothole_app']
                    collection = db.pothole_reports
                    
                    report_data = recommendation_report.copy()
                    
                    existing = collection.find_one({"image_filename": report_data["image_filename"]})
                    
                    if existing:
                        result = collection.update_one(
                            {"image_filename": report_data["image_filename"]},
                            {
                                "$inc": {
                                    "pothole_count": report_data["pothole_count"],
                                    "total_alerts": 1,
                                    "total_volume_cm3": report_data.get("total_volume_cm3", 0)
                                },
                                "$set": {
                                    "last_updated": datetime.datetime.now(datetime.timezone.utc),
                                    "alert_sent": True
                                },
                                "$push": {
                                    "detection_history": {
                                        "timestamp": datetime.datetime.now(datetime.timezone.utc),
                                        "potholes": report_data["pothole_count"],
                                        "image": report_data["image_filename"]
                                    }
                                }
                            }
                        )
                        print(f"📈 INCREMENTED! +{report_data['pothole_count']} potholes (now {existing['pothole_count'] + report_data['pothole_count']})")
                    else:
                        collection.insert_one(report_data)
                        print("✅ NEW report saved!")
                    
                    client.close()
                except Exception as e:
                    print(f"❌ MongoDB error: {e}")


        else:
            print("✅ No potholes detected.")
            flash("✅ No potholes detected in this image.", "success")


    return render_template(
        "dashboard.html",
        user_name=session.get("user_name"),
        detections=detections_table,
        annotated_image=annotated_rel_path
    )

@app.route('/save_detection', methods=['POST'])
def save_detection():
    address = request.form.get('address', 'Vepagunta Road')
    potholes = int(request.form.get('potholes_detected', 0))
    
    new_report = {
        'created_at': datetime.now().isoformat(),
        'place_name': address,
        'lat': float(request.form.get('lat', 17.6868)),
        'lon': float(request.form.get('lon', 83.1827)),
        'image_url': request.form.get('image_url', '/static/pothole1.jpg'),
        'bboxes': [[0.45, 0.35, 0.25, 0.30]],  
        'pothole_count': potholes,
        'total_volume_cm3': potholes * 3500,  
        'severity': 'Medium',
        'total_cost_min': potholes * 1000,
        'total_cost_max': potholes * 1600,
        'total_time_min': potholes * 20
    }
    
    if 'recommendation_reports' not in globals():
        globals()['recommendation_reports'] = []
    globals()['recommendation_reports'].append(new_report)
    
    try:
        from pymongo import MongoClient
        client = MongoClient('mongodb://localhost:27017/')
        db = client['pothole_app']
        db.pothole_reports.insert_one(new_report)
        print("✅ SAVED /save_detection TO MONGODB!")
        client.close()
    except Exception as e:
        print(f"MongoDB save_detection error: {e}")
    
    return jsonify({
        'success': True,
        'potholes': potholes,
        'address': address
    })



@app.route("/map", methods=["GET", "POST"])
@login_required
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
        if current_start and current_end:
            typed_start = current_start.get("name", "") or ""
            typed_end = current_end.get("name", "") or ""
            s_lat = current_start.get("lat")
            s_lon = current_start.get("lon")
            e_lat = current_end.get("lat")
            e_lon = current_end.get("lon")
            if s_lat is not None and s_lon is not None and e_lat is not None and e_lon is not None:
                route_geojson, route_source, route_error = get_route_geojson(s_lat, s_lon, e_lat, e_lon)

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

    NEAR_THRESHOLD_M = 300  
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

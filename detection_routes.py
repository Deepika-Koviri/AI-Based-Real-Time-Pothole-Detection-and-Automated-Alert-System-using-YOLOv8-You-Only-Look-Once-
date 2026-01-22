# detection_routes.py
from flask import (
    Blueprint,
    render_template,
    request,
    redirect,
    url_for,
    flash,
)
from flask_login import login_required, current_user
from bson.objectid import ObjectId
from datetime import datetime
import os
import cv2
import math
import requests

# from app_flask import mongo
from app import mongo  # Now works with blueprint setup

from map_utils import load_pothole_points

main_bp = Blueprint("main", __name__)
UPLOAD_ROOT = os.path.join("static", "uploads")

OSRM_URL = (
    "http://router.project-osrm.org/route/v1/driving/"
    "{lon1},{lat1};{lon2},{lat2}?overview=full&geometries=geojson"
)


def save_uploaded_file(file_storage):
    os.makedirs(UPLOAD_ROOT, exist_ok=True)
    ts = int(datetime.utcnow().timestamp())
    filename = f"{ts}_{file_storage.filename}"
    path = os.path.join(UPLOAD_ROOT, filename)
    file_storage.save(path)
    return path, filename


@main_bp.route("/")
def home():
    return redirect(url_for("auth.login"))


@main_bp.route("/dashboard", methods=["GET", "POST"])
@login_required
def dashboard():
    if request.method == "POST":
        input_type = request.form.get("input_type", "upload")
        img_file = request.files.get("image")

        if input_type == "upload" and not img_file:
            flash("Please upload an image", "warning")
            return redirect(url_for("main.dashboard"))

        rate_min = float(request.form.get("rate_min", 4000))
        rate_max = float(request.form.get("rate_max", 6000))
        max_depth_cm = float(request.form.get("max_depth_cm", 10))
        pixel_to_cm = float(request.form.get("pixel_to_cm", 0.5))

        address = request.form.get("address") or ""
        lat = request.form.get("lat")
        lon = request.form.get("lon")
        lat_val = float(lat) if lat else None
        lon_val = float(lon) if lon else None

        img_path, original_name = save_uploaded_file(img_file)

        img_doc = {
            "file_path": img_path,
            "original_filename": original_name,
            "source_type": input_type,
            "uploaded_by": ObjectId(current_user.id),
            "uploaded_at": datetime.utcnow(),
            "lat": lat_val,
            "lon": lon_val,
            "address": address,
        }
        img_id = mongo.db.images.insert_one(img_doc).inserted_id

        detections, annotated_bgr = run_pothole_detection(
            img_path,
            rate_min=rate_min,
            rate_max=rate_max,
            max_depth_cm=max_depth_cm,
            pixel_to_cm=pixel_to_cm,
            address=address,
            lat=lat_val,
            lon=lon_val,
        )

        ann_name = f"ann_{original_name}"
        ann_path = os.path.join(UPLOAD_ROOT, ann_name)
        cv2.imwrite(ann_path, annotated_bgr)

        for det in detections:
            pothole_id = mongo.db.potholes.insert_one(
                {"current_version": None}
            ).inserted_id

            vdoc = {
                "pothole_id": pothole_id,
                "version": 1,
                "image_id": img_id,
                "detected_by": ObjectId(current_user.id),
                "severity": det["Severity"],
                "area_cm2": det["Area_cm2"],
                "volume_cm3": det["Volume_cm3"],
                "cost_min": det["Cost_Min"],
                "cost_max": det["Cost_Max"],
                "time_min": det["Time_min"],
                "status": "active",
                "detected_at": datetime.utcnow(),
                "repaired_at": None,
            }
            v_id = mongo.db.pothole_versions.insert_one(vdoc).inserted_id

            mongo.db.potholes.update_one(
                {"_id": pothole_id},
                {"$set": {"current_version": v_id}},
            )

        rel_ann = ann_path.replace("static/", "")

        flash(f"Saved {len(detections)} potholes", "success")
        return render_template(
            "dashboard.html",
            user_name=current_user.name,
            detections=detections,
            annotated_image=rel_ann,
        )

    return render_template(
        "dashboard.html", user_name=current_user.name, detections=None
    )


@main_bp.route("/potholes/<pothole_id>/repaired", methods=["POST"])
@login_required
def mark_repaired(pothole_id):
    pothole_id = ObjectId(pothole_id)
    pothole = mongo.db.potholes.find_one({"_id": pothole_id})
    if not pothole:
        flash("Pothole not found", "danger")
        return redirect(url_for("main.dashboard"))

    current = mongo.db.pothole_versions.find_one(
        {"_id": pothole["current_version"]}
    )

    new_no = current["version"] + 1
    vdoc = {
        "pothole_id": pothole_id,
        "version": new_no,
        "image_id": current["image_id"],
        "detected_by": current["detected_by"],
        "severity": current["severity"],
        "area_cm2": current["area_cm2"],
        "volume_cm3": current["volume_cm3"],
        "cost_min": current["cost_min"],
        "cost_max": current["cost_max"],
        "time_min": current["time_min"],
        "status": "repaired",
        "detected_at": current["detected_at"],
        "repaired_at": datetime.utcnow(),
    }
    new_id = mongo.db.pothole_versions.insert_one(vdoc).inserted_id

    mongo.db.potholes.update_one(
        {"_id": pothole_id},
        {"$set": {"current_version": new_id}},
    )

    flash("Marked pothole as repaired", "success")
    return redirect(url_for("main.dashboard"))


# ---------- helpers for map ----------


def haversine(lat1, lon1, lat2, lon2):
    R = 6371000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = (
        math.sin(dphi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    )
    return 2 * R * math.asin(math.sqrt(a))


def filter_potholes_along_route(route_coords, potholes, max_dist_m=100):
    """Keep potholes whose distance to any route vertex <= max_dist_m."""
    filtered = []
    for p in potholes:
        plat, plon = p["lat"], p["lon"]
        for lon, lat in route_coords:  # route vertices are [lon, lat]
            if haversine(lat, lon, plat, plon) <= max_dist_m:
                filtered.append(p)
                break
    return filtered


@main_bp.route("/map", methods=["GET", "POST"])
def pothole_map():
    route_geojson = None
    potholes_on_route = []

    if request.method == "POST":
        # TODO: you can geocode start_location/end_location to lat/lon.
        # For now assume coordinates are already provided as hidden inputs.
        try:
            start_lat = float(request.form["start_lat"])
            start_lon = float(request.form["start_lon"])
            end_lat = float(request.form["end_lat"])
            end_lon = float(request.form["end_lon"])
        except (KeyError, ValueError):
            flash("Missing or invalid coordinates", "warning")
            return redirect(url_for("main.pothole_map"))

        url = OSRM_URL.format(
            lon1=start_lon, lat1=start_lat, lon2=end_lon, lat2=end_lat
        )
        r = requests.get(url)
        data = r.json()
        route_geojson = data["routes"][0]["geometry"]  # GeoJSON LineString

        all_potholes = load_pothole_points()
        potholes_on_route = filter_potholes_along_route(
            route_geojson["coordinates"], all_potholes, max_dist_m=100
        )

    return render_template(
        "map.html",
        route_geojson=route_geojson,
        potholes=potholes_on_route,
    )

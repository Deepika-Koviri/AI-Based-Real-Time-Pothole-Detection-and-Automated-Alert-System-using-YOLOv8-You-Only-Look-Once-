# map_routes.py
from flask import Blueprint, render_template, request, redirect, url_for, flash
import math
import requests
from map_utils import load_pothole_points

map_bp = Blueprint("map_bp", __name__)

# put your OpenRouteService API key here
ORS_API_KEY = "eyJvcmciOiI1YjNjZTM1OTc4NTExMTAwMDFjZjYyNDgiLCJpZCI6IjExZDgzMTFiMzhiYzQ1MTY4OWE2MTEyZTAwMTI3ZDgyIiwiaCI6Im11cm11cjY0In0="
ORS_URL = "https://api.openrouteservice.org/v2/directions/driving-car"


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
    filtered = []
    for p in potholes:
        plat, plon = p["lat"], p["lon"]
        for lon, lat in route_coords:  # ORS coords are [lon, lat]
            if haversine(lat, lon, plat, plon) <= max_dist_m:
                filtered.append(p)
                break
    return filtered


@map_bp.route("/map", methods=["GET", "POST"])
def pothole_map():
    route_geojson = None
    potholes_on_route = []

    if request.method == "POST":
        try:
            start_lat = float(request.form["start_lat"])
            start_lon = float(request.form["start_lon"])
            end_lat = float(request.form["end_lat"])
            end_lon = float(request.form["end_lon"])
        except (KeyError, ValueError):
            flash("Please pick start and destination on the map.", "warning")
            return redirect(url_for("map_bp.pothole_map"))

        body = {
            "coordinates": [
                [start_lon, start_lat],
                [end_lon, end_lat],
            ]
        }
        headers = {
            "Authorization": ORS_API_KEY,
            "Content-Type": "application/json",
        }
        r = requests.post(ORS_URL, json=body, headers=headers)
        data = r.json()

        # ORS returns GeoJSON FeatureCollection
        route_geojson = data["features"][0]["geometry"]

        all_potholes = load_pothole_points()
        potholes_on_route = filter_potholes_along_route(
            route_geojson["coordinates"], all_potholes, max_dist_m=100
        )

    return render_template(
        "map.html",
        route_geojson=route_geojson,
        potholes=potholes_on_route,
    )

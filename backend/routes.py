from flask import Blueprint, request, jsonify
from backend.extensions import db, socketio
from backend.models import AlertLog, User, EmergencyContact
from backend.alert_service import trigger_alert
from datetime import datetime, timedelta
from collections import Counter
from ai_model.person_detector import process_frame

routes_bp = Blueprint("routes_bp", __name__)

@routes_bp.route("/")
def home():
    return jsonify({"message": "SAFENET backend is working ✅"})

# ------------------- AUTH -------------------
@routes_bp.route("/register", methods=["POST"])
def register():
    data = request.get_json()
    name = data.get("name")
    email = data.get("email")
    password = data.get("password")
    phone = data.get("phone")

    if not name or not email or not password:
        return jsonify({"message": "Missing fields"}), 400

    if User.query.filter_by(email=email).first():
        return jsonify({"message": "Email already exists"}), 400

    user = User(name=name, email=email, phone=phone)
    user.set_password(password)
    db.session.add(user)
    db.session.commit()
    return jsonify({"message": "Registration successful"}), 201

@routes_bp.route("/login", methods=["POST"])
def login():
    data = request.get_json()
    email = data.get("email")
    password = data.get("password")
    user = User.query.filter_by(email=email).first()
    if user and user.check_password(password):
        return jsonify({"message": "Login successful", "user_id": user.id}), 200
    return jsonify({"message": "Invalid email or password"}), 401

# ------------------- ALERTS -------------------
@routes_bp.route("/send-alert", methods=["POST"])
def send_alert():
    data = request.get_json()
    alert_type = data.get("type")
    location = data.get("location")
    details = data.get("details", "")

    if not alert_type or not location:
        return jsonify({"error": "Missing data"}), 400

    alert = AlertLog(alert_type=alert_type, location=location, message=details)
    db.session.add(alert)
    db.session.commit()

    socketio.emit("new_alert", {
        "type": alert_type,
        "location": location,
        "message": details,
        "timestamp": alert.timestamp.strftime("%Y-%m-%d %H:%M:%S")
    })

    trigger_alert(alert_type=alert_type.lower(), location=location, details=details)
    return jsonify({"message": "Alert sent and notifications triggered!"})

@routes_bp.route("/alerts", methods=["GET"])
def get_alerts():
    alerts = AlertLog.query.order_by(AlertLog.timestamp.desc()).all()
    return jsonify([
        {
            "id": a.id,
            "type": a.alert_type,
            "location": a.location,
            "message": a.message,
            "timestamp": a.timestamp.strftime("%Y-%m-%d %H:%M:%S")
        } for a in alerts
    ])

# ------------------- HOTSPOTS -------------------
@routes_bp.route("/hotspots", methods=["GET"])
def get_hotspots():
    time_threshold = datetime.utcnow() - timedelta(minutes=15)
    recent_alerts = AlertLog.query.filter(AlertLog.timestamp >= time_threshold).all()
    locations = [a.location for a in recent_alerts]
    counts = Counter(locations)
    HOTSPOT_THRESHOLD = 3
    hotspots = {loc: count for loc, count in counts.items() if count >= HOTSPOT_THRESHOLD}
    return jsonify({"hotspots": hotspots})

# ------------------- VIDEO FRAME -------------------
@routes_bp.route("/video-frame", methods=["POST"])
def receive_frame():
    frame_file = request.files.get("frame")
    lat = request.form.get("lat", "Unknown")
    lon = request.form.get("lon", "Unknown")
    if frame_file:
        frame_bytes = frame_file.read()
        alert_generated = process_frame(frame_bytes, location=f"{lat},{lon}")
        return jsonify({"alert": alert_generated})
    return jsonify({"error": "No frame received"}), 400

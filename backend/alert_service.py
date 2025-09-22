from backend.models import AlertLog, User, EmergencyContact
from backend.extensions import db
from backend.utils.notifications import send_whatsapp_alert, send_sms_alert, send_voice_alert

def get_alert_receivers(alert_type, location=None, user_id=None):
    receivers = []

    if alert_type in ["anomaly", "hotspot"]:
        admins = User.query.filter_by(role="admin", alerts_enabled=True).all()
        receivers = [a.phone for a in admins if a.phone]

    elif alert_type == "sos" and user_id:
        contacts = EmergencyContact.query.filter_by(user_id=user_id).all()
        receivers = [c.phone for c in contacts if c.phone]

    return receivers


def trigger_alert(alert_type, location, user_id=None, details=""):
    message = f"⚠️ SAFENET Alert!\nType: {alert_type}\nLocation: {location}\nDetails: {details}"
    receivers = get_alert_receivers(alert_type, location, user_id)

    if not receivers:
        print("⚠️ No receivers found for this alert")
        return

    send_whatsapp_alert(message, receivers)
    send_sms_alert(message, receivers)
    send_voice_alert(message, receivers)

    new_alert = AlertLog(
        alert_type=alert_type,
        location=location,
        message=details,
        user_id=user_id
    )
    db.session.add(new_alert)
    db.session.commit()
    print(f"✅ Alert logged with ID {new_alert.id}")

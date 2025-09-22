import os
from twilio.rest import Client
from dotenv import load_dotenv

load_dotenv()

account_sid = os.getenv("TWILIO_ACCOUNT_SID")
auth_token = os.getenv("TWILIO_AUTH_TOKEN")
whatsapp_number = os.getenv("TWILIO_WHATSAPP_NUMBER")
twilio_phone = os.getenv("TWILIO_PHONE_NUMBER")

client = Client(account_sid, auth_token)

# -------- WhatsApp --------
def send_whatsapp_alert(message, receivers):
    for num in receivers:
        try:
            client.messages.create(
                from_=whatsapp_number,
                body=message,
                to=f"whatsapp:{num}"
            )
            print(f"✅ WhatsApp alert sent to {num}")
        except Exception as e:
            print(f"❌ WhatsApp failed for {num}: {e}")

# -------- SMS --------
def send_sms_alert(message, receivers):
    for num in receivers:
        try:
            client.messages.create(
                from_=twilio_phone,
                body=message,
                to=num
            )
            print(f"✅ SMS alert sent to {num}")
        except Exception as e:
            print(f"❌ SMS failed for {num}: {e}")

# -------- Voice Call (Text-to-Speech) --------
def send_voice_alert(message, receivers):
    for num in receivers:
        try:
            call = client.calls.create(
                twiml=f'<Response><Say>{message}</Say></Response>',
                to=num,
                from_=twilio_phone
            )
            print(f"✅ Voice call initiated to {num}, Call SID: {call.sid}")
        except Exception as e:
            print(f"❌ Voice call failed for {num}: {e}")

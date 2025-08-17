import os
import io
import uuid
import requests
import tempfile
from fastapi import FastAPI, Form
from fastapi.responses import FileResponse, JSONResponse
from twilio.rest import Client
from dotenv import load_dotenv

from summarise import summarise_clone_and_replay  # your pipeline

load_dotenv()
app = FastAPI()

# Twilio creds
account_sid = os.getenv("TWILIO_ACCOUNT_SID")
auth_token = os.getenv("TWILIO_AUTH_TOKEN")
twilio_whatsapp = "whatsapp:+14155238886"
client = Client(account_sid, auth_token)

# Public Render URL
PUBLIC_URL = None


@app.post("/whatsapp")
async def whatsapp_webhook(
    From: str = Form(...),
    MediaUrl0: str = Form(None)
):
    if not MediaUrl0:
        return JSONResponse({"error": "No media file received"}, status_code=400)

    # Download WhatsApp audio into memory
    r = requests.get(MediaUrl0)
    input_audio = io.BytesIO(r.content)

    # Run pipeline (returns raw bytes)
    output_audio_bytes = summarise_clone_and_replay(input_audio)

    # Save to temporary file for Twilio
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    temp_file.write(output_audio_bytes)
    temp_file.close()

    # Create public URL for Twilio to fetch
    file_name = os.path.basename(temp_file.name)
    file_url = f"{PUBLIC_URL}/temp_audio/{file_name}"

    # Send voice note back via WhatsApp
    client.messages.create(
        from_=twilio_whatsapp,
        to=From,
        media_url=[file_url]
    )

    return "OK"


@app.get("/temp_audio/{file_name}")
async def get_temp_audio(file_name: str):
    temp_dir = tempfile.gettempdir()
    file_path = os.path.join(temp_dir, file_name)

    if not os.path.exists(file_path):
        return JSONResponse({"error": "File not found"}, status_code=404)

    return FileResponse(file_path, media_type="audio/mpeg")

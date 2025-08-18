# app.py
import os
import io
import uuid
import asyncio
import requests
from fastapi import FastAPI, Form
from fastapi.responses import Response
from twilio.rest import Client
from dotenv import load_dotenv

from summarise import summarise_clone_and_replay  # your pipeline

load_dotenv()
app = FastAPI()

# Twilio creds from .env
account_sid = os.getenv("TWILIO_ACCOUNT_SID")
auth_token = os.getenv("TWILIO_AUTH_TOKEN")
twilio_whatsapp = "whatsapp:+14155238886"  # Twilio sandbox number

client = Client(account_sid, auth_token)

# In-memory store for audio blobs {id: (bytes, task)}
memory_store = {}


@app.post("/whatsapp")
async def whatsapp_webhook(
    From: str = Form(...),
    MediaUrl0: str = Form(None)  # Twilio sends voice notes here
):
    if not MediaUrl0:
        return "No media file received"

    # Download WhatsApp audio into memory
    r = requests.get(MediaUrl0, auth=(account_sid, auth_token))
    input_audio = r.content  # raw bytes

    # Print content-type header from Twilio
    print("DEBUG Twilio Content-Type:", r.headers.get("Content-Type"))
    print("DEBUG status code:", r.status_code)
    print(f"DEBUG (account_sid, auth_token): {(account_sid, auth_token)}")

    # Print first 32 bytes as hex to inspect file signature
    print("DEBUG first 32 bytes:", input_audio[:32].hex())

    # Run pipeline (returns MP3 bytes)
    output_audio = summarise_clone_and_replay(input_audio)

    # Store audio in memory with unique ID
    file_id = str(uuid.uuid4())
    memory_store[file_id] = output_audio

    # Schedule auto-cleanup after 5 minutes
    asyncio.create_task(_expire_file(file_id, delay=300))

    # Public URL for Twilio to fetch
    base_url = os.getenv("RENDER_EXTERNAL_URL")  # Render auto-sets this
    file_url = f"{base_url}/audio/{file_id}"

    # Send media back via WhatsApp
    client.messages.create(
        from_=twilio_whatsapp,
        to=From,
        media_url=[file_url]
    )

    return "OK"


@app.get("/audio/{file_id}")
async def get_audio(file_id: str):
    audio_bytes = memory_store.get(file_id)
    if not audio_bytes:
        return {"error": "File not found"}

    return Response(
        content=audio_bytes,
        media_type="audio/mpeg"  # mp3 is what ElevenLabs returns
    )


# --- auto-expire helper ---
async def _expire_file(file_id: str, delay: int = 300):
    await asyncio.sleep(delay)
    memory_store.pop(file_id, None)

import os
import io
import json
from typing import Optional

import requests
from openai import OpenAI

SYSTEM_PROMPT = """
You are a WhatsApp voice note transcribing AI. Your job is to read a transcription from a voice note, find the 4-5 most
important details, and share these in short sentences, but using the same style as the original message. E.g.
if someone sent a long voice note about starting a new job, being really excited about their new colleagues, talked
about hating their long commute, gave a big long update about their mother's surgery, and finally asked if you wanted to
meet for coffee next week, the summary should be something like:

Hey! New job is great and the people are lovely (commute is a bit of a pain though). Mum's doing well after the surgery.
Want to meet for coffee next week?

The tone should be friendly, upbeat but concise and ideally about 3 or 4 short sentences along with a greeting at the beginning
if they offered one originally. If they use particular turns of phrase (oh my god, for real) replicating those would be
good if they an be worked in naturally. If it's a really long message, going to 5 sentences to ensure you capture all of
the content is fine.
"""

# ---- Config ----
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is not set.")
if not ELEVENLABS_API_KEY:
    raise RuntimeError("ELEVENLABS_API_KEY is not set.")

client = OpenAI(api_key=OPENAI_API_KEY)

def _detect_audio_type(audio_bytes: bytes) -> str:
    header = audio_bytes[:16]
    if header.startswith(b"OggS"):
        return "ogg/opus"
    elif header[4:8] == b"ftyp":
        return "mp4/m4a"
    elif header[:2] == b"ID":
        return "mp3"
    else:
        return "unknown"



def _transcribe_audio(audio_bytes: bytes) -> str:
    """
    Send raw WhatsApp audio bytes (ogg/opus) directly to Whisper.
    """
    audio_io = io.BytesIO(audio_bytes)
    audio_io.name = "audio.ogg"  # helps SDK detect format
    audio_io.seek(0)

    resp = client.audio.transcriptions.create(
        model="whisper-1",
        file=audio_io
    )
    print(resp.text)
    return resp.text


def _summarise_text(text: str) -> str:
    """
    Summarise the transcript in first person, 2–3 sentences.
    """
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    SYSTEM_PROMPT
                )
            },
            {"role": "user", "content": text}
        ],
        temperature=0.3,
    )
    print(resp.choices[0].message.content.strip())
    return resp.choices[0].message.content.strip()


def _elevenlabs_clone_and_tts(summary_text: str, voice_sample_bytes: bytes) -> bytes:
    """
    Create a temporary ElevenLabs voice from the sample, then TTS the summary.
    Returns MP3 bytes.
    """
    # 1) Create temporary voice
    add_voice_url = "https://api.elevenlabs.io/v1/voices/add"

    files = {
        # filename helps server infer content type/format; ogg/opus is fine
        "files": ("sample.ogg", voice_sample_bytes)
    }
    data = {"name": "temp-voice"}

    res = requests.post(
        add_voice_url,
        headers={"xi-api-key": ELEVENLABS_API_KEY},
        files=files,
        data=data,
        timeout=60,
    )

    try:
        payload = res.json()
    except Exception:
        payload = {}
    if not res.ok or "voice_id" not in payload:
        raise RuntimeError(f"Voice cloning failed: HTTP {res.status_code} {res.text}")

    voice_id = payload["voice_id"]

    # 2) TTS using that temporary voice
    tts_url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"

    body = {
        "text": summary_text,
        "model_id": "eleven_multilingual_v2",
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 1,
            "style": 0.5,
            "use_speaker_boost": True
        }
    }

    tts_res = requests.post(
        tts_url,
        headers={
            "xi-api-key": ELEVENLABS_API_KEY,
            "Content-Type": "application/json",
            "Accept": "audio/mpeg",  # ask for MP3
        },
        data=json.dumps(body),
        timeout=120,
    )
    if not tts_res.ok or not tts_res.content:
        raise RuntimeError(f"TTS failed: HTTP {tts_res.status_code} {tts_res.text}")

    return tts_res.content  # MP3 bytes


def summarise_clone_and_replay(input_audio_bytes: bytes, filename: Optional[str] = None) -> bytes:
    """
    End-to-end:
      1) Transcribe with Whisper
      2) Summarise with GPT
      3) Clone the voice and speak summary with ElevenLabs
    Returns MP3 bytes ready to send back to WhatsApp.
    """

    print("DEBUG audio type:", _detect_audio_type(input_audio_bytes))

    # Transcribe directly (no ffmpeg conversion)
    transcript = _transcribe_audio(input_audio_bytes)

    # Summarise
    summary = _summarise_text(transcript)

    # Clone & TTS (we pass the ORIGINAL sample bytes to preserve accent)
    mp3_bytes = _elevenlabs_clone_and_tts(summary, input_audio_bytes)

    return mp3_bytes

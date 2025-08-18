# summarise.py
import os
import io
import requests
from pydub import AudioSegment
import imageio_ffmpeg
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")

# 1. Transcribe audio
def transcribe_audio(input_audio_bytesio):
    # Convert incoming audio (ogg/opus) to WAV in memory
    audio = AudioSegment.from_file(input_audio_bytesio)  # autodetect format
    out_io = io.BytesIO()
    audio.export(out_io, format="wav")
    out_io.seek(0)

    transcript = client.audio.transcriptions.create(
        model="whisper-1",
        file=out_io
    )
    return transcript.text

# 2. Summarise text
def summarise_text(text):
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": """
            Rewrite the following voice message in a single paragraph. Keep the style and speech of the original and keep it in the first person.
            Keep it to 2-3 sentences MAX.
            """},
            {"role": "user", "content": text}
        ]
    )
    return resp.choices[0].message.content

# 3. Clone voice with ElevenLabs
def clone_voice_and_replay(summary_text, voice_sample_bytesio):
    # Upload sample to ElevenLabs to create a temporary voice
    voice_url = "https://api.elevenlabs.io/v1/voices/add"

    res = requests.post(
        voice_url,
        headers={"xi-api-key": ELEVENLABS_API_KEY},
        files={"files": voice_sample_bytesio},
        data={"name": "temp-voice"}
    )

    if res.ok and "voice_id" in res.json():
        voice_id = res.json()["voice_id"]
    else:
        raise RuntimeError(f"Voice cloning failed: {res.text}")

    tts_url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"

    payload = {
        "text": summary_text,
        "model_id": "eleven_multilingual_v2",
        "voice_settings": {
            "stability": 0.6,
            "similarity_boost": 1.0,
            "style": 0.0,
            "use_speaker_boost": True
        }
    }

    audio_res = requests.post(
        tts_url,
        headers={
            "xi-api-key": ELEVENLABS_API_KEY,
            "Content-Type": "application/json"
        },
        json=payload
    )

    return io.BytesIO(audio_res.content)  # return as BytesIO

# Full pipeline
def summarise_clone_and_replay(input_audio_bytesio):
    transcript = transcribe_audio(input_audio_bytesio)
    summary_text = summarise_text(transcript)
    output_audio_bytesio = clone_voice_and_replay(summary_text, input_audio_bytesio)
    return output_audio_bytesio

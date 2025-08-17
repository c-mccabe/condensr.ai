import os
import requests
from pydub import AudioSegment
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")

# 1. Transcribe audio
def transcribe_audio(file_path):
    with open(file_path, "rb") as f:
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=f
        )
    return transcript.text

# 2. Summarise text
def summarise_text(text):
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": """
            Rewrite the following voice message in a single paragraph. Keep the style and speech of the original and keep it in the first person.
            Keep it to 2-3 sentences MAX. Please keep it all in the first person.
            """},
            {"role": "user", "content": text}
        ]
    )
    return resp.choices[0].message.content

# 3. Clone voice with ElevenLabs
def clone_voice(summary_text, voice_sample_path, output_path):
    # Upload sample to ElevenLabs to create a temporary voice
    voice_url = "https://api.elevenlabs.io/v1/voices/add"

    with open(voice_sample_path, "rb") as sample:
        res = requests.post(
            voice_url,
            headers={"xi-api-key": ELEVENLABS_API_KEY},
            files={"files": sample},
            data={"name": "temp-voice"}
        )

    print(res.status_code, res.text)  # 🔍 See the raw API response

    if res.ok and "voice_id" in res.json():
        voice_id = res.json()["voice_id"]
    else:
        raise RuntimeError(f"Voice cloning failed: {res.text}")

    tts_url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"

    payload = {
        "text": summary_text,
        "model_id": "eleven_multilingual_v2",  # better at accents
        "voice_settings": {
            "stability": 0.6,  # natural pacing
            "similarity_boost": 1.0,  # maximum voice matching
            "style": 0.0,  # no extra stylisation
            "use_speaker_boost": True  # keep timbre strong
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

    with open(output_path, "wb") as f:
        f.write(audio_res.content)

# 4. Convert MP3/WAV → OGG for WhatsApp
def convert_to_ogg(input_path, output_path):
    sound = AudioSegment.from_file(input_path)
    sound.export(output_path, format="ogg", codec="libopus")
    

if __name__ == "__main__":
    input_file = "raw_audio/ogg/conor_example.ogg"
    transcript = transcribe_audio(input_file)
    summary = summarise_text(transcript)
    print("Summary:", summary)

    clone_voice(summary, input_file, "summary_clone.mp3")
    convert_to_ogg("summary_clone.mp3", "summary_clone.ogg")
    print("Saved WhatsApp-ready OGG file.")

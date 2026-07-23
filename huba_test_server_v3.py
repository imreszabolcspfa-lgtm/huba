import os
from flask import Flask, Response, send_from_directory
from google.cloud import texttospeech
import google.generativeai as genai

app = Flask(__name__)

AUDIO_DIR = "static_audio"
os.makedirs(AUDIO_DIR, exist_ok=True)

# Configure Gemini using the API key stored as an environment variable
# (set with $env:GEMINI_API_KEY = "..." before running this server)
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

# This is Huba's persona -- kept short for now, we'll expand this
# properly once the basic loop is proven to work.
HUBA_SYSTEM_PROMPT = """
Te Huba vagy, egy segítőkész, barátságos telefonos asszisztens, aki
természetes, hétköznapi magyar nyelven beszél. Gyakorlati kérdésekben
segítesz -- otthoni javítás, főzés, hivatalos ügyek, technológia.

Fontos szabályok:
- Rövid, érthető válaszokat adj, mintha egy szomszédnak magyaráznál.
- Ha a kérdés veszélyes lehet (pl. elektromos vagy gázszerelés), javasold,
  hogy kérjen szakember segítségét.
- Sosem adj orvosi diagnózist vagy jogi képviseletet.
- Legyél őszinte, ha valamiben nem vagy biztos.
"""

model = genai.GenerativeModel(
    model_name="gemini-2.0-flash",
    system_instruction=HUBA_SYSTEM_PROMPT,
)


def ask_huba(question):
    """Sends a question to Gemini and returns Huba's text response."""
    response = model.generate_content(question)
    return response.text


def generate_hungarian_audio(text, filename):
    client = texttospeech.TextToSpeechClient()
    synthesis_input = texttospeech.SynthesisInput(text=text)
    voice = texttospeech.VoiceSelectionParams(
        language_code="hu-HU",
        ssml_gender=texttospeech.SsmlVoiceGender.MALE,
    )
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3
    )
    response = client.synthesize_speech(
        input=synthesis_input, voice=voice, audio_config=audio_config
    )
    filepath = os.path.join(AUDIO_DIR, filename)
    with open(filepath, "wb") as out:
        out.write(response.audio_content)
    return filepath


@app.route("/voice", methods=["POST"])
def voice():
    # TEMPORARY: a hardcoded test question, standing in for what Whisper
    # will eventually provide once real caller audio is wired in.
    test_question = "Hogyan tudom megjavítani egy csöpögő csapot?"

    huba_reply = ask_huba(test_question)
    print("Huba mondta:", huba_reply)  # so you can see it in the terminal too

    generate_hungarian_audio(huba_reply, "reply.mp3")

    audio_url = f"https://{__import__('flask').request.host}/audio/reply.mp3"

    twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Play>{audio_url}</Play>
</Response>"""
    return Response(twiml, mimetype="text/xml")


@app.route("/audio/<filename>")
def serve_audio(filename):
    return send_from_directory(AUDIO_DIR, filename)


if __name__ == "__main__":
    app.run(port=5000)

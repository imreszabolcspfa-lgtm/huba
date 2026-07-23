import os
from flask import Flask, Response, send_from_directory
from google.cloud import texttospeech

app = Flask(__name__)

# Folder where we'll save generated audio files so Twilio can fetch them
AUDIO_DIR = "static_audio"
os.makedirs(AUDIO_DIR, exist_ok=True)


def generate_hungarian_audio(text, filename):
    """Calls Google Cloud TTS and saves the result as an mp3 file."""
    client = texttospeech.TextToSpeechClient()

    synthesis_input = texttospeech.SynthesisInput(text=text)

    # Instead of hardcoding a specific voice name (which was defaulting
    # to a female voice), we specify the language and gender and let
    # Google pick a matching Hungarian male voice automatically.
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
    text = "Szia! Itt Huba. A rendszer működik."

    # Generate the audio file fresh each call (fine for testing;
    # later we'd cache common phrases instead of regenerating every time)
    generate_hungarian_audio(text, "greeting.mp3")

    # Twilio needs a public URL to fetch the audio from — this builds
    # one using whatever host the request came in on (your ngrok URL)
    audio_url = f"https://{__import__('flask').request.host}/audio/greeting.mp3"

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

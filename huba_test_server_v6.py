import json
import base64
from flask import Flask, Response
from flask_sock import Sock
import audioop

app = Flask(__name__)
sock = Sock(app)

# These two numbers are the "tuning knobs" for pause detection --
# we may need to adjust them once we hear how it behaves in practice.
SILENCE_THRESHOLD = 500        # amplitude below this counts as silence
SILENCE_CHUNKS_FOR_PAUSE = 25  # ~25 chunks x 20ms = ~0.5s of silence = "pause"


@app.route("/voice", methods=["POST"])
def voice():
    twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Connect>
        <Stream url="wss://{__import__('flask').request.host}/media-stream" />
    </Connect>
</Response>"""
    return Response(twiml, mimetype="text/xml")


@sock.route("/media-stream")
def media_stream(ws):
    print(">>> Media stream connected")

    audio_buffer = bytearray()
    silence_run = 0
    speaking = False

    while True:
        message = ws.receive()
        if message is None:
            print(">>> Media stream closed")
            break

        data = json.loads(message)
        event = data.get("event")

        if event == "connected":
            print(">>> Twilio confirmed connection")

        elif event == "start":
            print(">>> Call stream starting:", data["start"]["callSid"])

        elif event == "media":
            # Twilio sends 8kHz mulaw-encoded audio; convert to 16-bit PCM
            # so we can actually measure its volume.
            mulaw_bytes = base64.b64decode(data["media"]["payload"])
            pcm_bytes = audioop.ulaw2lin(mulaw_bytes, 2)
            amplitude = audioop.rms(pcm_bytes, 2)

            if amplitude > SILENCE_THRESHOLD:
                # Caller is actively talking in this chunk
                audio_buffer.extend(pcm_bytes)
                silence_run = 0
                if not speaking:
                    print(">>> Speech started")
                    speaking = True
            else:
                if speaking:
                    silence_run += 1
                    audio_buffer.extend(pcm_bytes)  # keep a little trailing silence
                    if silence_run >= SILENCE_CHUNKS_FOR_PAUSE:
                        seconds = len(audio_buffer) / 2 / 8000
                        print(f">>> Pause detected -- buffered {seconds:.2f}s "
                              f"of audio ({len(audio_buffer)} bytes). "
                              f"[Next step: send this to Whisper]")
                        audio_buffer = bytearray()
                        speaking = False
                        silence_run = 0

        elif event == "stop":
            print(">>> Call stream stopped.")


if __name__ == "__main__":
    app.run(port=5000)

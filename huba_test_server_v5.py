import os
import json
from flask import Flask, Response
from flask_sock import Sock

app = Flask(__name__)
sock = Sock(app)


@app.route("/voice", methods=["POST"])
def voice():
    # Instead of <Play> (one fixed audio file), <Connect><Stream> opens
    # a live, ongoing audio connection for the whole call, to the
    # WebSocket route defined below.
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

    chunk_count = 0

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
            chunk_count += 1
            payload_length = len(data["media"]["payload"])
            # Only print every 25th chunk so the terminal doesn't flood --
            # each chunk is ~20ms of audio, so this is roughly every 0.5s
            if chunk_count % 25 == 0:
                print(f">>> Received audio chunk #{chunk_count} "
                      f"({payload_length} bytes, base64-encoded)")

        elif event == "stop":
            print(">>> Call stream stopped. Total chunks received:",
                  chunk_count)


if __name__ == "__main__":
    app.run(port=5000)

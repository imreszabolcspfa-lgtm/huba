from flask import Flask, Response

app = Flask(__name__)

@app.route("/voice", methods=["POST"])
def voice():
    # This is the simplest possible response: Twilio calls this URL
    # when someone dials your number, and reads back whatever TwiML
    # (Twilio's XML instruction format) we return.
    twiml = """<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say language="hu-HU">Szia! Itt Huba. A rendszer működik.</Say>
</Response>"""
    return Response(twiml, mimetype="text/xml")

if __name__ == "__main__":
    # Runs locally on port 5000. ngrok will expose this to the internet.
    app.run(port=5000)

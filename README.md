# Huba

Huba is a Hungarian-language phone advisory service: a caller dials a number and talks to an AI assistant (Székely tone) that answers everyday questions — broken devices, official forms, cooking and preserving, phone setup. No app, internet or registration needed. It is aimed mainly at older people and anyone who finds it easier to ask out loud than to type a search.

Huba is not a doctor, lawyer or crisis line, and says so. It opens every call by stating it is an AI.

Landing page: https://huba-landing.vercel.app/

## Layout

- `public/` — static landing page (deployed on Vercel)
- `huba_server.py` — the voice server (Flask + websocket)
- `huba_compliance_roadmap.pdf` — GDPR / Romanian telecom / EU AI Act to-do list
- Earlier prototypes (v1–v11) are in the git history.

## How a call works

1. Twilio receives the call and POSTs to `/voice`, which answers with TwiML opening a websocket to `/media-stream`.
2. Caller audio (8 kHz mulaw) is streamed to Deepgram (`nova-3`) for live transcription.
3. About 1 s of silence ends the caller's turn.
4. Gemini writes a short Hungarian reply (Google Search grounding, conversation memory for the whole call).
5. edge-tts (`hu-HU-TamasNeural`) speaks it back sentence by sentence while the rest is still generating; short filler phrases cover the wait.

## Running locally

Python 3.13 needs `audioop-lts` (included in `requirements.txt`); `ffmpeg` must be on the PATH for pydub.

```powershell
pip install -r requirements.txt
$env:DEEPGRAM_API_KEY = "..."
$env:GEMINI_API_KEY = "..."
python huba_server.py
```

Expose the server (e.g. with ngrok) and set the Twilio number's "A call comes in" webhook to `https://<host>/voice` (HTTP POST).

## Notes

- Keys are read from environment variables only; never commit credentials (`*.json` is git-ignored).
- The server writes caller audio chunks to `debug_audio/`. They are not deleted automatically — see the compliance roadmap before taking real callers.

## Deploying (always-on)

A phone number needs a server that is always reachable, not a laptop behind ngrok.

1. Create a Render account, choose **New → Blueprint**, and select this repo (`render.yaml` and the `Dockerfile` do the rest). Use a paid instance — free ones sleep when idle.
2. In the service's **Environment** tab set `DEEPGRAM_API_KEY` and `GEMINI_API_KEY`.
3. When the deploy is live (`https://<name>.onrender.com/health` returns `ok`), set the Twilio number's **A call comes in** webhook to `https://<name>.onrender.com/voice` (HTTP POST).

Any host that runs a Docker container and supports websockets (Fly.io, Railway) works the same way.

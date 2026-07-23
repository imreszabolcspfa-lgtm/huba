import json
import base64
import os
import wave
import io
import time
import asyncio
from collections import deque
import requests
from flask import Flask, Response
from flask_sock import Sock
import audioop
import google.generativeai as genai
import edge_tts
from pydub import AudioSegment

app = Flask(__name__)
sock = Sock(app)

# Debug aid: dump every buffered chunk to a .wav file right before it goes
# out for transcription, so we can actually listen to what's being sent
# instead of guessing from the transcript alone.
DEBUG_AUDIO_DIR = "debug_audio"
os.makedirs(DEBUG_AUDIO_DIR, exist_ok=True)
_debug_chunk_counter = 0


def save_debug_wav(pcm_8k_bytes: bytes) -> str:
    global _debug_chunk_counter
    _debug_chunk_counter += 1
    path = os.path.join(DEBUG_AUDIO_DIR, f"chunk_{_debug_chunk_counter:04d}.wav")
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)   # 16-bit PCM
        wf.setframerate(8000)  # original Twilio rate
        wf.writeframes(pcm_8k_bytes)
    return path


# These are the "tuning knobs" for pause detection, arrived at through
# real testing in v7 (see comments below for why each value is what it is).
SILENCE_THRESHOLD = 300        # amplitude below this counts as silence
SILENCE_CHUNKS_FOR_PAUSE = 75  # ~75 chunks x 20ms = ~1.5s of silence = "pause"

# New in v11: local Whisper (large-v3, CPU) is gone. It was accurate but
# structurally slow -- 8-9s per utterance with no GPU to run on. Deepgram's
# Nova-3 model runs the equivalent work on their GPUs and streams the
# result back, which should turn that 8-9s into roughly 1-2s. Requires
# DEEPGRAM_API_KEY as an environment variable.
DEEPGRAM_API_KEY = os.environ.get("DEEPGRAM_API_KEY")
if not DEEPGRAM_API_KEY:
    raise RuntimeError(
        "DEEPGRAM_API_KEY environment variable is not set. "
        "Set it before running, e.g. in PowerShell: $env:DEEPGRAM_API_KEY = \"your-key-here\""
    )
DEEPGRAM_URL = "https://api.deepgram.com/v1/listen"

# Gemini generates Huba's actual reply text from the transcript. Requires
# GEMINI_API_KEY to be set as an environment variable -- never hardcode
# the key directly in the file.
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY environment variable is not set. "
        "Set it before running, e.g. in PowerShell: $env:GEMINI_API_KEY = \"your-key-here\""
    )
genai.configure(api_key=GEMINI_API_KEY)

# Model name is a config knob -- swap this if Google renames/deprecates it.
GEMINI_MODEL_NAME = "gemini-3.5-flash"
gemini_model = genai.GenerativeModel(
    GEMINI_MODEL_NAME,
    system_instruction=(
        "A neved Huba, egy telefonos hangalapú asszisztens vagy. "
        "Mindig magyarul válaszolj, természetesen és tömören -- "
        "ahogy egy telefonbeszélgetésben beszélnénk, nem írásban. "
        "Ne legyél túl hosszú: egy-két mondat elég."
    ),
)
print(f">>> Gemini model '{GEMINI_MODEL_NAME}' ready.")

# edge-tts (Microsoft's free neural TTS, no API key) gives us a named male
# Hungarian voice, unlike gTTS which only has one voice per language.
EDGE_TTS_VOICE = "hu-HU-TamasNeural"  # male. Female alternative: hu-HU-NoemiNeural


def transcribe_buffer(mulaw_8k_bytes: bytes) -> str:
    """Send raw 8kHz mulaw audio (Twilio's native format -- no conversion
    needed) to Deepgram and return the Hungarian transcript."""
    params = {
        "language": "hu",
        "model": "nova-3",
        "smart_format": "true",
        "encoding": "mulaw",
        "sample_rate": "8000",
        "channels": "1",
    }
    headers = {
        "Authorization": f"Token {DEEPGRAM_API_KEY}",
        "Content-Type": "audio/mulaw",
    }
    response = requests.post(
        DEEPGRAM_URL, params=params, headers=headers, data=mulaw_8k_bytes, timeout=15
    )
    response.raise_for_status()
    result = response.json()
    return result["results"]["channels"][0]["alternatives"][0]["transcript"].strip()


def generate_reply(transcript: str) -> str:
    """Send the transcript to Gemini and return Huba's Hungarian reply text."""
    response = gemini_model.generate_content(transcript)
    return response.text.strip()


async def _edge_tts_to_mp3_bytes(text: str) -> bytes:
    communicate = edge_tts.Communicate(text, EDGE_TTS_VOICE)
    mp3_buffer = io.BytesIO()
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            mp3_buffer.write(chunk["data"])
    return mp3_buffer.getvalue()


def synthesize_speech(text: str) -> bytes:
    """Turn reply text into 8kHz mulaw bytes -- the exact format Twilio's
    Media Streams expects for outbound audio."""
    mp3_bytes = asyncio.run(_edge_tts_to_mp3_bytes(text))

    audio = AudioSegment.from_file(io.BytesIO(mp3_bytes), format="mp3")
    audio = audio.set_frame_rate(8000).set_channels(1).set_sample_width(2)
    pcm_bytes = audio.raw_data

    return audioop.lin2ulaw(pcm_bytes, 2)


def send_audio_to_twilio(ws, stream_sid: str, mulaw_bytes: bytes):
    """Stream mulaw audio back to the caller in 20ms frames (Twilio's
    native chunk size -- 160 bytes at 8kHz, 1 byte/sample for mulaw)."""
    frame_size = 160
    for i in range(0, len(mulaw_bytes), frame_size):
        frame = mulaw_bytes[i:i + frame_size]
        message = {
            "event": "media",
            "streamSid": stream_sid,
            "media": {"payload": base64.b64encode(frame).decode("ascii")},
        }
        ws.send(json.dumps(message))
        time.sleep(0.02)  # pace roughly at real-time (20ms per frame)


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

    audio_buffer = bytearray()      # linear PCM -- used for VAD + debug wav
    mulaw_buffer = bytearray()      # raw mulaw -- sent to Deepgram directly
    silence_run = 0
    speaking = False
    stream_sid = None

    # Rolling pre-roll: always holds the last ~300ms of raw audio (both
    # forms), speech or not, so we don't lose a quiet-onset word at the
    # start of an utterance.
    PRE_ROLL_CHUNKS = 15  # ~15 x 20ms = ~300ms
    pre_roll = deque(maxlen=PRE_ROLL_CHUNKS)

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
            stream_sid = data["start"]["streamSid"]
            print(">>> Call stream starting:", data["start"]["callSid"])

        elif event == "media":
            # Twilio sends 8kHz mulaw-encoded audio. Keep the raw mulaw
            # (for Deepgram) and a linear-PCM copy (for volume measurement
            # and debug wavs).
            mulaw_bytes = base64.b64decode(data["media"]["payload"])
            pcm_bytes = audioop.ulaw2lin(mulaw_bytes, 2)
            amplitude = audioop.rms(pcm_bytes, 2)

            if not speaking:
                pre_roll.append((pcm_bytes, mulaw_bytes))

            if amplitude > SILENCE_THRESHOLD:
                # Caller is actively talking in this chunk
                if not speaking:
                    # Recover the ~300ms leading into this moment, in case
                    # the very start of the word was quieter than threshold.
                    for pre_pcm, pre_mulaw in pre_roll:
                        audio_buffer.extend(pre_pcm)
                        mulaw_buffer.extend(pre_mulaw)
                    pre_roll.clear()
                    print(">>> Speech started")
                    speaking = True
                audio_buffer.extend(pcm_bytes)
                mulaw_buffer.extend(mulaw_bytes)
                silence_run = 0
            else:
                if speaking:
                    silence_run += 1
                    audio_buffer.extend(pcm_bytes)  # keep a little trailing silence
                    mulaw_buffer.extend(mulaw_bytes)
                    if silence_run >= SILENCE_CHUNKS_FOR_PAUSE:
                        seconds = len(audio_buffer) / 2 / 8000
                        print(f">>> Pause detected -- buffered {seconds:.2f}s "
                              f"of audio ({len(audio_buffer)} bytes). Transcribing...")

                        wav_path = save_debug_wav(bytes(audio_buffer))
                        print(f">>> Saved debug audio: {wav_path}")

                        t0 = time.perf_counter()
                        text = transcribe_buffer(bytes(mulaw_buffer))
                        t1 = time.perf_counter()
                        print(f">>> Deepgram heard: \"{text}\"  ({t1 - t0:.2f}s)")

                        if text:
                            reply = generate_reply(text)
                            t2 = time.perf_counter()
                            print(f">>> Huba replies: \"{reply}\"  ({t2 - t1:.2f}s)")

                            mulaw_reply = synthesize_speech(reply)
                            t3 = time.perf_counter()
                            print(f">>> Speech synthesized ({len(mulaw_reply)} bytes)  ({t3 - t2:.2f}s)")

                            send_audio_to_twilio(ws, stream_sid, mulaw_reply)
                            t4 = time.perf_counter()
                            print(f">>> Done speaking. (send took {t4 - t3:.2f}s, "
                                  f"total turnaround {t4 - t0:.2f}s)")
                        else:
                            print(">>> (empty transcript, skipping Gemini call)")

                        audio_buffer = bytearray()
                        mulaw_buffer = bytearray()
                        speaking = False
                        silence_run = 0

        elif event == "stop":
            print(">>> Call stream stopped.")


if __name__ == "__main__":
    app.run(port=5000)

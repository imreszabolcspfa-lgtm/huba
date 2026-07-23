import json
import base64
import os
import wave
from collections import deque
import numpy as np
from flask import Flask, Response
from flask_sock import Sock
import audioop
from faster_whisper import WhisperModel
import google.generativeai as genai
 
app = Flask(__name__)
sock = Sock(app)
 
# Debug aid: dump every buffered chunk to a .wav file right before it goes
# to Whisper, so we can actually listen to what the model is being fed
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
        wf.setframerate(8000)  # original Twilio rate, pre-resample
        wf.writeframes(pcm_8k_bytes)
    return path
 
 
# These are the "tuning knobs" for pause detection, arrived at through
# real testing in v7 (see comments below for why each value is what it is).
SILENCE_THRESHOLD = 300        # amplitude below this counts as silence
SILENCE_CHUNKS_FOR_PAUSE = 75  # ~75 chunks x 20ms = ~1.5s of silence = "pause"
 
WHISPER_MODEL_SIZE = "large-v3"
 
print(">>> Loading Whisper model (this happens once, at startup)...")
whisper_model = WhisperModel(WHISPER_MODEL_SIZE, device="cpu", compute_type="int8")
print(">>> Whisper model loaded.")
 
# New in v8: Gemini generates Huba's actual reply text from the transcript.
# Requires GEMINI_API_KEY to be set as an environment variable -- never
# hardcode the key directly in the file.
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
 
 
def transcribe_buffer(pcm_8k_bytes: bytes) -> str:
    """Take raw 16-bit PCM audio at 8kHz (Twilio's rate) and return
    Hungarian text. Whisper expects 16kHz float32 audio, so we resample
    first."""
    resampled, _ = audioop.ratecv(pcm_8k_bytes, 2, 1, 8000, 16000, None)
    audio_np = np.frombuffer(resampled, dtype=np.int16).astype(np.float32) / 32768.0
 
    segments, _info = whisper_model.transcribe(
        audio_np,
        language="hu",
        beam_size=10,
        vad_filter=True,
        condition_on_previous_text=False,
        initial_prompt="Ez egy hétköznapi, természetes magyar nyelvű beszélgetés telefonon.",
    )
    return "".join(segment.text for segment in segments).strip()
 
 
def generate_reply(transcript: str) -> str:
    """Send the transcript to Gemini and return Huba's Hungarian reply text.
    This is text only for now -- turning it into speech is the next step."""
    response = gemini_model.generate_content(transcript)
    return response.text.strip()
 
 
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
 
    # Rolling pre-roll: always holds the last ~300ms of raw audio, speech or
    # not, so we don't lose a quiet-onset word at the start of an utterance.
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
            print(">>> Call stream starting:", data["start"]["callSid"])
 
        elif event == "media":
            # Twilio sends 8kHz mulaw-encoded audio; convert to 16-bit PCM
            # so we can actually measure its volume.
            mulaw_bytes = base64.b64decode(data["media"]["payload"])
            pcm_bytes = audioop.ulaw2lin(mulaw_bytes, 2)
            amplitude = audioop.rms(pcm_bytes, 2)
 
            if not speaking:
                pre_roll.append(pcm_bytes)
 
            if amplitude > SILENCE_THRESHOLD:
                # Caller is actively talking in this chunk
                if not speaking:
                    # Recover the ~300ms leading into this moment, in case
                    # the very start of the word was quieter than threshold.
                    for pre_chunk in pre_roll:
                        audio_buffer.extend(pre_chunk)
                    pre_roll.clear()
                    print(">>> Speech started")
                    speaking = True
                audio_buffer.extend(pcm_bytes)
                silence_run = 0
            else:
                if speaking:
                    silence_run += 1
                    audio_buffer.extend(pcm_bytes)  # keep a little trailing silence
                    if silence_run >= SILENCE_CHUNKS_FOR_PAUSE:
                        seconds = len(audio_buffer) / 2 / 8000
                        print(f">>> Pause detected -- buffered {seconds:.2f}s "
                              f"of audio ({len(audio_buffer)} bytes). Transcribing...")
 
                        wav_path = save_debug_wav(bytes(audio_buffer))
                        print(f">>> Saved debug audio: {wav_path}")
 
                        text = transcribe_buffer(bytes(audio_buffer))
                        print(f">>> Whisper heard: \"{text}\"")
 
                        if text:
                            reply = generate_reply(text)
                            print(f">>> Huba replies: \"{reply}\"")
                        else:
                            print(">>> (empty transcript, skipping Gemini call)")
 
                        audio_buffer = bytearray()
                        speaking = False
                        silence_run = 0
 
        elif event == "stop":
            print(">>> Call stream stopped.")
 
 
if __name__ == "__main__":
    app.run(port=5000)
 
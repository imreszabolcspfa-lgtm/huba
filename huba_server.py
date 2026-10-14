import json
import base64
import os
import re
import wave
import io
import time
import asyncio
import threading
import queue
import random
from collections import deque
from flask import Flask, Response, request
from flask_sock import Sock
from simple_websocket import ConnectionClosed
import audioop
from google import genai
from google.genai import types as genai_types
import edge_tts
from pydub import AudioSegment
from deepgram import DeepgramClient
from deepgram.core.events import EventType

app = Flask(__name__)
sock = Sock(app)

# Debug aid: dump every buffered chunk to a .wav file for offline listening.
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
# real testing in v7. They're still used here -- not to gate transcription
# anymore (Deepgram is now transcribing continuously in the background),
# but purely to decide "the caller has finished their turn, wrap it up."
SILENCE_THRESHOLD = 300        # amplitude below this counts as silence
# Was 75 (~1.5s), tightened to shave latency off every response. Trade-off:
# a full 1s mid-sentence pause could now get treated as end-of-turn. If
# Huba starts jumping in before you're done talking, bump this back up.
SILENCE_CHUNKS_FOR_PAUSE = 50  # ~50 chunks x 20ms = ~1.0s of silence = "pause"

# v11's Deepgram REST call buffered the whole utterance locally before
# sending it as one blob (5s, not the sub-300ms Deepgram is built for).
# This version streams audio to Deepgram continuously as it arrives from
# Twilio, so transcription happens in parallel with you talking.
DEEPGRAM_API_KEY = os.environ.get("DEEPGRAM_API_KEY")
if not DEEPGRAM_API_KEY:
    raise RuntimeError(
        "DEEPGRAM_API_KEY environment variable is not set. "
        "Set it before running, e.g. in PowerShell: $env:DEEPGRAM_API_KEY = \"your-key-here\""
    )
deepgram_client = DeepgramClient(api_key=DEEPGRAM_API_KEY)

# Gemini generates Huba's actual reply text from the transcript. Requires
# GEMINI_API_KEY to be set as an environment variable -- never hardcode
# the key directly in the file.
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY environment variable is not set. "
        "Set it before running, e.g. in PowerShell: $env:GEMINI_API_KEY = \"your-key-here\""
    )
gemini_client = genai.Client(api_key=GEMINI_API_KEY)

GEMINI_MODEL_NAME = "gemini-3.5-flash"
# google_search enables real-time Google Search grounding: for queries like
# a company's phone number or address, Gemini can now issue an actual
# search and answer from real results instead of guessing from stale
# training data. Trade-off: adds real latency on exactly the queries that
# trigger a search (not every turn) -- if that's noticeable, lean on the
# existing filler-phrase system to cover it rather than removing this.
#
# Note: this requires the current `google-genai` SDK (import as
# `from google import genai`), not the deprecated `google-generativeai`
# package. That older SDK's tool builder only accepts the literal string
# "code_execution" (or the older, now-unsupported-for-this-model
# "google_search_retrieval") as a bare string -- it has no object-based way
# to express the newer "google_search" tool at all.
GEMINI_CONFIG = genai_types.GenerateContentConfig(
    system_instruction=(
        "A neved Huba, egy telefonos hangalapú asszisztens vagy. "
        "Mindig magyarul válaszolj, természetesen és tömören -- "
        "ahogy egy telefonbeszélgetésben beszélnénk, nem írásban. "
        "Ne legyél túl hosszú: egy-két mondat elég."
    ),
    tools=[genai_types.Tool(google_search=genai_types.GoogleSearch())],
)
print(f">>> Gemini client ready, model '{GEMINI_MODEL_NAME}'.")

# edge-tts (Microsoft's free neural TTS, no API key) gives us a named male
# Hungarian voice, unlike gTTS which only has one voice per language.
EDGE_TTS_VOICE = "hu-HU-TamasNeural"  # male. Female alternative: hu-HU-NoemiNeural

# Trust/warmth tuning: default Tamás read as bright and brisk -- more
# "customer service script" than "someone you'd trust with something
# personal." Slightly slower and slightly lower-pitched reads as calmer
# and more grounded. These are starting values, not a final answer --
# worth listening and nudging further (e.g. -12% rate, -8Hz pitch) if
# it's still not quite right.
EDGE_TTS_RATE = "-7%"    # negative = slower
EDGE_TTS_PITCH = "-8Hz"  # negative = lower


SENTENCE_END_RE = re.compile(r'[.!?]+(\s|$)')


def stream_reply_sentences(chat_session, transcript: str):
    """Stream Gemini's reply and yield each sentence as soon as it's
    complete, instead of waiting for the whole (possibly multi-sentence)
    reply to finish generating. This lets us start synthesizing and
    speaking the first sentence while Gemini is still generating the rest,
    instead of doing the two steps fully sequentially.

    Takes a chat_session (not the bare model) so each call adds to a
    running conversation history -- without this, every reply was
    generated with zero memory of anything said earlier in the call."""
    buffer = ""
    response = chat_session.send_message_stream(transcript)
    for chunk in response:
        buffer += getattr(chunk, "text", "") or ""
        while True:
            match = SENTENCE_END_RE.search(buffer)
            if not match:
                break
            sentence = buffer[:match.end()].strip()
            buffer = buffer[match.end():]
            if sentence:
                yield sentence
    remainder = buffer.strip()
    if remainder:
        yield remainder


_STREAM_DONE = object()


def start_reply_audio_stream(chat_session, transcript: str):
    """Start generating Huba's reply (Gemini + per-sentence TTS) in a
    background thread RIGHT NOW, and return an iterator to consume the
    results as they arrive.

    This is a plain function, not a generator -- unlike a generator, its
    body (including starting the thread) runs immediately when called,
    not lazily on first iteration. That matters here: we want Gemini/TTS
    working in the background *while* a filler phrase plays, not only
    starting once we begin pulling sentences out."""
    q = queue.Queue(maxsize=1)

    def producer():
        try:
            for sentence in stream_reply_sentences(chat_session, transcript):
                t_start = time.perf_counter()
                mulaw_chunk = synthesize_speech(sentence)
                q.put((sentence, mulaw_chunk, time.perf_counter() - t_start))
        finally:
            q.put(_STREAM_DONE)

    threading.Thread(target=producer, daemon=True).start()

    def consume():
        while True:
            item = q.get()
            if item is _STREAM_DONE:
                break
            yield item

    return consume()


async def _edge_tts_to_mp3_bytes(text: str) -> bytes:
    communicate = edge_tts.Communicate(
        text, EDGE_TTS_VOICE, rate=EDGE_TTS_RATE, pitch=EDGE_TTS_PITCH,
    )
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


# Filler phrases: acknowledge the caller immediately (near-zero delay, since
# these are pre-synthesized once at startup, not generated per-turn) while
# Gemini + TTS work on the real reply in the background. Masks the
# unavoidable network latency instead of making the caller sit in silence.
FILLER_PHRASES = [
    "Hamarosan mondom.",
    "Nemsokára szólok.",
    "Rövidesen válaszolok.",
    "Rövid időn belül mondom.",
    "Nem tart sokáig, szólok.",
    "Egy kis időn belül szólok.",
    "Nem kell sokat várnod, mondom.",
    "Hamarosan válaszolok rá.",
    "Rövidesen mondom is.",
    "Nem tart sokáig, válaszolok rá.",
]

# Short, neutral filler used in the other half of turns, as a plainer
# alternative to the longer phrases above.
FILLER_SHORT_PHRASE = "Rendben."

print(">>> Pre-synthesizing filler phrases...")
FILLER_AUDIO = [(phrase, synthesize_speech(phrase)) for phrase in FILLER_PHRASES]
FILLER_SHORT_AUDIO = (FILLER_SHORT_PHRASE, synthesize_speech(FILLER_SHORT_PHRASE))
print(f">>> {len(FILLER_AUDIO)} filler phrases ready (plus the short '{FILLER_SHORT_PHRASE}' fallback).")


def pick_filler_audio():
    """50% of turns: one of the longer, varied filler phrases. The other
    50%: just "Rendben." -- keeps the acknowledgment from feeling
    repetitive without needing more phrases than we can meaningfully vary."""
    if random.random() < 0.5:
        return random.choice(FILLER_AUDIO)
    return FILLER_SHORT_AUDIO

# Opening greeting, played once at the start of every call. "mesterséges
# intelligenciára épülő" makes it unambiguous this is an AI, not just an
# automated phone system -- satisfies the EU AI Act's plain-disclosure
# requirement while keeping the light "no hold music" joke.
GREETING_TEXT = (
    "Üdvözöllek, Huba vagyok, egy mesterséges intelligencia, a telefonos "
    "asszisztensed -- máris itt vagyok, élőben, mondd, miben segíthetek."
)
print(">>> Pre-synthesizing greeting...")
GREETING_AUDIO = synthesize_speech(GREETING_TEXT)
print(">>> Greeting ready.")


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
        try:
            ws.send(json.dumps(message))
        except ConnectionClosed:
            # Caller hung up mid-reply: stop playing instead of crashing.
            return
        time.sleep(0.02)  # pace roughly at real-time (20ms per frame)


def keep_deepgram_alive_during(dg_connection, blocking_fn):
    """Run blocking_fn() while pinging Deepgram in the background so it
    doesn't time out (NET-0001) during stretches where we're not forwarding
    real caller audio -- e.g. while Huba's reply is being played back, which
    can take several seconds and would otherwise starve the connection."""
    stop = threading.Event()

    def ping_loop():
        while not stop.wait(3):
            try:
                dg_connection.send_keep_alive()
            except Exception:
                break

    pinger = threading.Thread(target=ping_loop, daemon=True)
    pinger.start()
    try:
        blocking_fn()
    finally:
        stop.set()


# GET is accepted too: a Twilio number whose webhook method is left on GET
# would otherwise get a 405 and the caller hears "application error".
@app.route("/voice", methods=["GET", "POST"])
def voice():
    # Behind ngrok / a reverse proxy the public hostname arrives in
    # X-Forwarded-Host; fall back to Host when running without a proxy.
    host = request.headers.get("X-Forwarded-Host") or request.host
    twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Connect>
        <Stream url="wss://{host}/media-stream" />
    </Connect>
</Response>"""
    return Response(twiml, mimetype="text/xml")


@app.route("/health")
def health():
    return "ok"


@sock.route("/media-stream")
def media_stream(ws):
    print(">>> Media stream connected")

    audio_buffer = bytearray()      # linear PCM -- used for VAD + debug wav
    silence_run = 0
    speaking = False
    stream_sid = None

    # One chat session per call, reused for every turn -- this is what
    # gives Huba actual memory of the conversation instead of treating
    # each reply as a fresh, context-free request.
    chat_session = gemini_client.chats.create(model=GEMINI_MODEL_NAME, config=GEMINI_CONFIG)

    # Accumulates Deepgram's finalized transcript pieces for the utterance
    # currently in progress. Deepgram sends these back asynchronously (via
    # the callback below) while audio is still streaming in.
    transcript_parts = []

    # Rolling pre-roll: always holds the last ~300ms of raw PCM, speech or
    # not, purely for the debug wav / VAD onset recovery -- Deepgram itself
    # receives every chunk regardless, so it doesn't need a pre-roll splice.
    PRE_ROLL_CHUNKS = 15  # ~15 x 20ms = ~300ms
    pre_roll = deque(maxlen=PRE_ROLL_CHUNKS)

    # Both an eager connection and a lazily-opened one hit the identical
    # instant NET-0001 failure, which ruled out "idle gap before first
    # audio" as the cause. Next suspect: calling __enter__()/__exit__()
    # manually instead of using connect() as a real `with` block -- this
    # SDK's background listener thread may depend on that scoping to wire
    # up correctly. Using the documented pattern exactly here.
    with deepgram_client.listen.v1.connect(
        model="nova-3",
        language="hu",
        smart_format="true",
        encoding="mulaw",
        sample_rate="8000",
        channels="1",
        interim_results="true",
    ) as dg_connection:

        def on_message(message):
            msg_type = getattr(message, "type", None)
            if msg_type not in ("Results", None):
                return
            try:
                alternative = message.channel.alternatives[0]
                sentence = alternative.transcript
            except (AttributeError, IndexError):
                return
            if not sentence:
                return
            if getattr(message, "is_final", False):
                transcript_parts.append(sentence)
                print(f">>> Deepgram (final piece): \"{sentence}\"")
            else:
                print(f">>> Deepgram (interim): \"{sentence}\"")

        dg_connection.on(EventType.MESSAGE, on_message)
        dg_connection.on(EventType.ERROR, lambda error: print(f">>> Deepgram error: {error}"))

        # ROOT CAUSE of the repeated NET-0001 failures: start_listening() is
        # a *blocking* read loop (confirmed from the SDK source), not a
        # background subscription. Calling it inline meant we never reached
        # the code that sends Twilio's audio until Deepgram had already
        # given up waiting and closed the socket. Running it in its own
        # thread frees the main thread to forward audio immediately.
        listener_thread = threading.Thread(target=dg_connection.start_listening, daemon=True)
        listener_thread.start()
        print(">>> Deepgram streaming connection open.")

        while True:
            try:
                message = ws.receive()
            except ConnectionClosed:
                message = None
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

                print(">>> Playing greeting...")
                keep_deepgram_alive_during(
                    dg_connection,
                    lambda: send_audio_to_twilio(ws, stream_sid, GREETING_AUDIO),
                )
                print(">>> Greeting done.")

            elif event == "media":
                mulaw_bytes = base64.b64decode(data["media"]["payload"])
                pcm_bytes = audioop.ulaw2lin(mulaw_bytes, 2)
                amplitude = audioop.rms(pcm_bytes, 2)

                # Forward every chunk to Deepgram immediately, regardless of
                # our own silence detection -- this is what lets
                # transcription happen in parallel with you talking.
                dg_connection.send_media(mulaw_bytes)

                if not speaking:
                    pre_roll.append(pcm_bytes)

                if amplitude > SILENCE_THRESHOLD:
                    if not speaking:
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
                        audio_buffer.extend(pcm_bytes)
                        if silence_run >= SILENCE_CHUNKS_FOR_PAUSE:
                            seconds = len(audio_buffer) / 2 / 8000
                            print(f">>> Pause detected -- {seconds:.2f}s of audio. "
                                  f"Wrapping up turn...")

                            wav_path = save_debug_wav(bytes(audio_buffer))
                            print(f">>> Saved debug audio: {wav_path}")

                            t0 = time.perf_counter()
                            # We already waited SILENCE_CHUNKS_FOR_PAUSE
                            # (1.5s) of local silence before even getting
                            # here -- Deepgram has had plenty of time to
                            # finalize the tail end by now, so this only
                            # needs to be a small safety margin, not the
                            # 0.4s we used to give it.
                            time.sleep(0.1)
                            text = " ".join(transcript_parts).strip()
                            transcript_parts.clear()
                            t1 = time.perf_counter()
                            print(f">>> Deepgram heard: \"{text}\"  ({t1 - t0:.2f}s)")

                            if text:
                                # Start Gemini + TTS working on the real
                                # reply RIGHT NOW, in the background --
                                # before we've even played anything. Then
                                # immediately play a quick filler phrase
                                # ("Egy pillanat...") to acknowledge the
                                # caller instead of leaving them in silence.
                                # By the time the filler finishes, the real
                                # first sentence is often already sitting
                                # ready. This only masks the wait -- it
                                # doesn't remove it -- but it's the
                                # difference between dead air and a
                                # responsive-feeling assistant.
                                reply_stream = start_reply_audio_stream(chat_session, text)
                                filler_phrase, filler_mulaw = pick_filler_audio()

                                def speak_full_reply():
                                    print(f">>> Huba (filler): \"{filler_phrase}\"")
                                    send_audio_to_twilio(ws, stream_sid, filler_mulaw)

                                    is_first = True
                                    for sentence, mulaw_chunk, synth_seconds in reply_stream:
                                        t_sentence = time.perf_counter()
                                        if is_first:
                                            print(f">>> Huba replies: \"{sentence}\"  "
                                                  f"(first sentence ready {t_sentence - t1:.2f}s after transcript, "
                                                  f"synth took {synth_seconds:.2f}s)")
                                            is_first = False
                                        else:
                                            print(f">>> Huba continues: \"{sentence}\"  "
                                                  f"(synth took {synth_seconds:.2f}s)")
                                        send_audio_to_twilio(ws, stream_sid, mulaw_chunk)

                                keep_deepgram_alive_during(dg_connection, speak_full_reply)

                                t4 = time.perf_counter()
                                print(f">>> Done speaking. (total turnaround {t4 - t0:.2f}s)")
                            else:
                                print(">>> (empty transcript, skipping Gemini call)")

                            audio_buffer = bytearray()
                            speaking = False
                            silence_run = 0

            elif event == "stop":
                print(">>> Call stream stopped.")
                dg_connection.send_close_stream()
                break


if __name__ == "__main__":
    # Local run only; in production gunicorn serves `huba_server:app` (see Dockerfile).
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
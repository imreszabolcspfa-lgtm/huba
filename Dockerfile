FROM python:3.12-slim

# pydub needs ffmpeg to convert edge-tts output to 8 kHz mulaw for Twilio.
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY huba_server.py .

# One worker (call state lives in-process), many threads (one per live call).
CMD gunicorn -w 1 -k gthread --threads 50 --timeout 120 -b 0.0.0.0:${PORT:-5000} huba_server:app

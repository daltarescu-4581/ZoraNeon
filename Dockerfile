FROM python:3.11-slim

# opencv-python-headless still links against glib even without the GUI bits.
# ca-certificates is needed to reach the Anthropic and Supabase APIs.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libglib2.0-0 ca-certificates \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies first so a code change does not re-download the wheels.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY fridge_watcher/ ./fridge_watcher/

# Logs are JSON lines on stdout and must not sit in a pipe buffer.
ENV PYTHONUNBUFFERED=1

CMD ["python", "-m", "fridge_watcher", "run"]

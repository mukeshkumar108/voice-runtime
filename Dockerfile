FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PATH="/usr/src/app/.venv/bin:${PATH}"
# All runtime configuration arrives via VOICE_* env vars (see .env.example).
ENV VOICE_PORT=3002

WORKDIR /usr/src/app

# Install packages
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        git \
        libportaudio2 \
        libsndfile1 \
        python3 \
        python3-pip \
        python3-venv \
    && rm -rf /var/lib/apt/lists/*
RUN python3 -m pip install --no-cache-dir --break-system-packages uv

COPY pyproject.toml README.md LICENSE MANIFEST.in ./
RUN uv sync --python /usr/bin/python3 --no-install-project --no-dev

COPY . .
RUN uv sync --python /usr/bin/python3 --no-dev
RUN python -c "import nltk; nltk.download('punkt_tab'); nltk.download('averaged_perceptron_tagger_eng')"

COPY docker/entrypoint.sh /usr/local/bin/voice-entrypoint.sh
RUN chmod +x /usr/local/bin/voice-entrypoint.sh

EXPOSE 3002

HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:' + __import__('os').environ.get('VOICE_PORT', '3002') + '/healthz')"

ENTRYPOINT ["/usr/local/bin/voice-entrypoint.sh"]

# Digital Music Store: Gradio web chat
#
# Build:  docker build -t digital-music-store .
# Run:    docker run --rm -p 7860:7860 --env-file .env digital-music-store
# Open:   http://localhost:7860
#
# The API key is passed at run time (--env-file); .dockerignore keeps .env
# out of the image.

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # Listen on all interfaces: 127.0.0.1 inside a container is unreachable
    # from the host, even with the port published.
    HOST=0.0.0.0 \
    PORT=7860 \
    # Don't send Gradio usage analytics from the container.
    GRADIO_ANALYTICS_ENABLED=False

WORKDIR /app

# Dependencies first, so code changes don't reinstall them.
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

# Don't run as root.
RUN useradd --create-home appuser
USER appuser

EXPOSE 7860

# Healthy once the page answers (standard library only: no curl in slim).
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\", \"7860\")}/', timeout=4)"

CMD ["python", "app.py"]

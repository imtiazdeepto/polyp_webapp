FROM python:3.11-slim

WORKDIR /app

# OpenCV runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY model.py inference.py backend.py ./

# Mount or copy best.pth at runtime for production deployments
ENV PORT=8000
ENV CHECKPOINT_PATH=best.pth

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/')"

CMD ["sh", "-c", "uvicorn backend:app --host 0.0.0.0 --port ${PORT}"]

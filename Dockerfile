# Образ сервера FRS PAD: Python 3.12, залежності прототипу, моделі ONNX з випуску репозиторію.
# Вебкамеру захоплює браузер, тому контейнеру доступ до камери не потрібен.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PAD_HOST=0.0.0.0 \
    PAD_PORT=8000 \
    PAD_DATA_DIR=/app/data \
    PAD_LOG_DIR=/logs

# бібліотеки, потрібні OpenCV і MediaPipe в образі без графічного середовища
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY pad ./pad
COPY scripts ./scripts
COPY configs ./configs
RUN python scripts/download_models.py --prebuilt

RUN useradd --create-home --uid 1000 frs \
    && mkdir -p /app/data /logs \
    && chown -R frs:frs /app/data /logs
USER frs

EXPOSE 8000
VOLUME ["/app/data", "/logs"]
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/')" || exit 1
CMD ["python", "scripts/serve.py", "--classifier", "models/pad_logistic.json"]

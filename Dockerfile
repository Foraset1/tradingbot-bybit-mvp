FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TRADINGBOT_CONFIG=/app/config/tradingbot.toml \
    TRADINGBOT_DATA_ROOT=/data/raw \
    TRADINGBOT_ARCHIVE_ROOT=/data/archive \
    TRADINGBOT_HISTORY_ROOT=/data/history

WORKDIR /app

RUN apt-get update \
    && apt-get install --no-install-recommends --yes libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src ./src
COPY config ./config

RUN pip install --no-cache-dir '.[research]'

RUN useradd --create-home --uid 10001 tradingbot \
    && mkdir -p /data/raw /data/archive /data/history /app/runtime \
    && chown -R tradingbot:tradingbot /data /app/runtime

USER tradingbot

VOLUME ["/data"]

CMD ["python", "-m", "tradingbot", "collect"]

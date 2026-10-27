"""Системний журнал сервера у форматі JSON (ТЗ, п. 4.2.4).

Одна подія на рядок у файлі logs/frs-pad.log; файл змінюється щодоби опівночі,
зберігаються останні 30 файлів. Рівень задається змінною середовища LOG_LEVEL
(типово INFO), каталог – PAD_LOG_DIR.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path

KEEP_DAYS = 30


class JsonFormatter(logging.Formatter):
    """Подія журналу як JSON-об'єкт: час, рівень, модуль, повідомлення і, якщо є, виняток."""

    def format(self, record: logging.LogRecord) -> str:
        entry = {"time": datetime.fromtimestamp(record.created).isoformat(timespec="milliseconds"),
                 "level": record.levelname, "logger": record.name, "message": record.getMessage()}
        if record.exc_info:
            entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(entry, ensure_ascii=False)


def setup_json_log(log_dir: str | Path | None = None) -> Path:
    """Додає до кореневого журналу файл JSON з щодобовою ротацією; повертає шлях до файлу."""
    path = Path(log_dir or os.environ.get("PAD_LOG_DIR", "logs")) / "frs-pad.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = TimedRotatingFileHandler(path, when="midnight", backupCount=KEEP_DAYS, encoding="utf-8")
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.addHandler(handler)
    root.setLevel(os.environ.get("LOG_LEVEL", "INFO").upper())
    return path

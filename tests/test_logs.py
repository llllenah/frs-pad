"""Системний журнал у форматі JSON (pad/logs.py, ТЗ п. 4.2.4)."""

import json
import logging

from pad.logs import KEEP_DAYS, JsonFormatter, setup_json_log


def test_json_line_per_event():
    record = logging.LogRecord("pad.server", logging.WARNING, __file__, 1, "відхилено %s", ("запит",), None)
    entry = json.loads(JsonFormatter().format(record))
    assert entry["level"] == "WARNING" and entry["logger"] == "pad.server"
    assert entry["message"] == "відхилено запит" and "T" in entry["time"]


def test_daily_rotation_keeps_30_files(tmp_path, monkeypatch):
    monkeypatch.setenv("LOG_LEVEL", "WARNING")
    root = logging.getLogger()
    before, level = list(root.handlers), root.level
    try:
        path = setup_json_log(tmp_path)
        handler = next(h for h in root.handlers if h not in before)
        assert handler.when == "MIDNIGHT" and handler.backupCount == KEEP_DAYS == 30
        assert root.level == logging.WARNING
        logging.getLogger("pad.test").warning("подія")
        handler.flush()
        assert json.loads(path.read_text(encoding="utf-8").splitlines()[-1])["message"] == "подія"
    finally:
        for h in root.handlers[:]:
            if h not in before:
                root.removeHandler(h)
                h.close()
        root.setLevel(level)

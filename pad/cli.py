"""Спільні налаштування для скриптів командного рядка."""

from __future__ import annotations

import argparse
import logging
import os
import sys

import cv2
import numpy as np

from .storage import DatasetKeyError, DecryptionError, load_embedding


def setup_console() -> None:
    """Вмикає UTF-8 у консолі Windows і налаштовує журналювання (рівень з LOG_LEVEL)."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"),
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    os.environ.setdefault("GLOG_minloglevel", "2")  # менше службових повідомлень MediaPipe
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


def open_source(args, cfg: dict) -> tuple[cv2.VideoCapture, float | None]:
    """Відкриває вебкамеру або відеофайл; для файлу повертає його частоту кадрів."""
    if args.video:
        cap = cv2.VideoCapture(args.video)
        if not cap.isOpened():
            sys.exit(f"не вдалося відкрити відео {args.video}")
        return cap, cap.get(cv2.CAP_PROP_FPS) or 30.0
    backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
    cap = cv2.VideoCapture(args.camera, backend)
    if not cap.isOpened():
        sys.exit(f"не вдалося відкрити камеру {args.camera}")
    cam = cfg.get("camera", {})
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width or cam.get("width", 1280))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height or cam.get("height", 720))
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap, None


def add_common_args(ap: argparse.ArgumentParser) -> None:
    """Спільні аргументи demo.py і record_session.py."""
    ap.add_argument("--config", default=None, help="YAML-конфігурація (configs/pad.yaml)")
    ap.add_argument("--camera", type=int, default=0, help="індекс камери")
    ap.add_argument("--video", default=None, help="відеофайл замість камери")
    ap.add_argument("--width", type=int, default=None)
    ap.add_argument("--height", type=int, default=None)
    ap.add_argument("--classifier", default=None, help="файл моделі .json або .onnx")
    ap.add_argument("--enrolled", default=None, help="ембединг заявленої особи (.npy.enc або .npy)")
    ap.add_argument("--headless", action="store_true", help="без вікна (для вимірювань)")
    ap.add_argument("--auto-start", action="store_true", help="почати сеанс з першого кадру")
    ap.add_argument("--max-frames", type=int, default=None)


def load_enrolled(path: str | None, key: bytes | None) -> np.ndarray | None:
    """Ембединг з --enrolled; для .npy.enc без ключа або з іншим ключем – вихід з поясненням."""
    if path is None:
        return None
    try:
        return load_embedding(path, key)
    except (DatasetKeyError, DecryptionError) as exc:
        sys.exit(f"не вдалося прочитати ембединг: {exc}")

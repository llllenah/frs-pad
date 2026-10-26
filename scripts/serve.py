"""Веб-інтерфейс прототипу: екранні форми 4.7 (сеанс верифікації) і 4.8 (запис сеансів).

Запускає сервер FastAPI з REST і WebSocket; браузер захоплює кадри вебкамери
і надсилає їх серверу, сервер обробляє їх тим самим конвеєром, що й demo.py.
Приклади:
    python scripts/serve.py                  # http://127.0.0.1:8000
    python scripts/serve.py --port 8080 --classifier models/pad_logistic.json

Ключ PAD_DATASET_KEY потрібен для форми запису сеансів (записи і еталони
зберігаються лише зашифрованими, FR-65). Без ключа працює форма верифікації,
а еталон, зареєстрований у браузері, лишається лише в пам'яті сервера.
Сервер слухає лише 127.0.0.1: getUserMedia у браузері дозволено для localhost
без HTTPS, а відкривати біометричний потік у мережу не варто.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pad.cli import setup_console  # noqa: E402
from pad.runtime import build_runtime, load_config, resolve  # noqa: E402
from pad.server import PadService, create_app  # noqa: E402
from pad.storage import DatasetKeyError, load_key_if_set  # noqa: E402


def main() -> int:
    setup_console()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None, help="YAML-конфігурація (configs/pad.yaml)")
    ap.add_argument("--classifier", default=None, help="файл моделі .json або .onnx")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--data", default=str(ROOT / "data"), help="тека з enrolled/ і recordings/")
    args = ap.parse_args()

    try:
        key = load_key_if_set()
    except DatasetKeyError as exc:
        sys.exit(str(exc))
    if key is None:
        print("PAD_DATASET_KEY не задано: форма запису сеансів недоступна, еталони лише в пам'яті")

    import uvicorn

    cfg = load_config(args.config)
    pipeline, manager = build_runtime(cfg, args.classifier)
    pipeline.warmup()
    clf_path = resolve(args.classifier or cfg.get("classifier"))
    app = create_app(PadService(pipeline, manager, args.data, key, cfg.get("monitoring"),
                                classifier_path=clf_path if clf_path is not None and clf_path.exists() else None))
    print(f"відкрийте http://{args.host}:{args.port}/ у Chrome або Edge")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    sys.exit(main())

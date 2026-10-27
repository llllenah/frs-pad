"""Веб-інтерфейс прототипу: екранні форми 4.7 (сеанс верифікації) і 4.8 (запис сеансів).

Запускає сервер FastAPI з REST і WebSocket; браузер захоплює кадри вебкамери
і надсилає їх серверу, сервер обробляє їх тим самим конвеєром, що й demo.py.
Приклади:
    python scripts/serve.py                  # http://127.0.0.1:8000
    python scripts/serve.py --port 8080 --classifier models/pad_logistic.json
    python scripts/serve.py --host 0.0.0.0 --ssl-certfile cert.pem --ssl-keyfile key.pem

Вхід до системи: облікові записи у <data>/users.json (scripts/manage_users.py).
Якщо записів ще немає, перший адміністратор створюється зі змінних середовища
PAD_ADMIN_EMAIL і PAD_ADMIN_PASSWORD. Ключ підпису токенів – PAD_JWT_SECRET
(без нього ключ випадковий і після перезапуску потрібно увійти знову).

Ключ PAD_DATASET_KEY потрібен для форми запису сеансів (записи і еталони
зберігаються лише зашифрованими, FR-65). Без ключа працює форма верифікації,
а еталон, зареєстрований у браузері, лишається лише в пам'яті сервера.
Типово сервер слухає лише 127.0.0.1: getUserMedia у браузері дозволено для
localhost без HTTPS. Для доступу з мережі потрібні --host і сертифікат TLS
(HTTPS і WSS), інакше біометричний потік іде відкритим текстом.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pad.auth import Auth, AuthError, UserStore  # noqa: E402
from pad.cli import setup_console  # noqa: E402
from pad.logs import setup_json_log  # noqa: E402
from pad.runtime import build_runtime, load_config, resolve  # noqa: E402
from pad.server import PadService, create_app, local_hosts  # noqa: E402
from pad.storage import DatasetKeyError, load_key_if_set  # noqa: E402

LOCAL = ("127.0.0.1", "localhost", "::1")


def build_auth(data_dir: Path) -> Auth:
    """Облікові записи і журнал аудиту в теці даних; перший адміністратор – зі змінних середовища."""
    store = UserStore(data_dir / "users.json")
    if not len(store):
        email, password = os.environ.get("PAD_ADMIN_EMAIL"), os.environ.get("PAD_ADMIN_PASSWORD")
        if not (email and password):
            sys.exit("немає жодного облікового запису: задайте PAD_ADMIN_EMAIL і PAD_ADMIN_PASSWORD "
                     "або створіть адміністратора командою\n"
                     "    python scripts/manage_users.py add EMAIL --role admin")
        try:
            store.create(email, password, "admin")
        except AuthError as exc:
            sys.exit(f"адміністратора не створено: {exc}")
        print(f"створено обліковий запис адміністратора {email.strip().lower()}")
    try:
        auth = Auth(store, os.environ.get("PAD_JWT_SECRET"), data_dir / "audit.jsonl")
    except AuthError as exc:
        sys.exit(f"PAD_JWT_SECRET: {exc}; згенеруйте ключ командою\n"
                 "    python -c \"import secrets; print(secrets.token_urlsafe(48))\"")
    if auth.ephemeral:
        print("PAD_JWT_SECRET не задано: ключ підпису токенів випадковий, після перезапуску потрібно увійти знову")
    return auth


def main() -> int:
    setup_console()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None, help="YAML-конфігурація (configs/pad.yaml)")
    ap.add_argument("--classifier", default=None, help="файл моделі .json або .onnx")
    ap.add_argument("--host", default=os.environ.get("PAD_HOST", "127.0.0.1"))
    ap.add_argument("--port", type=int, default=int(os.environ.get("PAD_PORT", 8000)))
    ap.add_argument("--data", default=os.environ.get("PAD_DATA_DIR", str(ROOT / "data")),
                    help="тека з enrolled/, recordings/, users.json")
    ap.add_argument("--ssl-certfile", default=os.environ.get("PAD_SSL_CERTFILE"), help="сертифікат TLS (PEM)")
    ap.add_argument("--ssl-keyfile", default=os.environ.get("PAD_SSL_KEYFILE"), help="закритий ключ TLS (PEM)")
    args = ap.parse_args()

    try:
        key = load_key_if_set()
    except DatasetKeyError as exc:
        sys.exit(str(exc))
    if key is None:
        print("PAD_DATASET_KEY не задано: форма запису сеансів недоступна, еталони лише в пам'яті")

    import uvicorn

    data_dir = Path(args.data)
    auth = build_auth(data_dir)
    log_path = setup_json_log()
    tls = bool(args.ssl_certfile and args.ssl_keyfile)

    cfg = load_config(args.config)
    pipeline, manager = build_runtime(cfg, args.classifier, max_hands=2)
    pipeline.warmup()
    clf_path = resolve(args.classifier or cfg.get("classifier"))
    extra_hosts = os.environ.get("PAD_ALLOWED_HOSTS", "").split(",")
    app = create_app(PadService(pipeline, manager, str(data_dir), key, cfg.get("monitoring"),
                                classifier_path=clf_path if clf_path is not None and clf_path.exists() else None),
                     allowed_hosts=local_hosts(args.host, args.port, extra_hosts), auth=auth, secure_cookies=tls)
    if args.host not in LOCAL and not tls:
        print(f"УВАГА: сервер доступний з мережі ({args.host}) без TLS; біометричний відеопотік іде без шифрування")
    scheme = "https" if tls else "http"
    shown = "localhost" if args.host in ("0.0.0.0", "::") else args.host
    print(f"журнал: {log_path}")
    print(f"відкрийте {scheme}://{shown}:{args.port}/ у Chrome або Edge")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning",
                ssl_certfile=args.ssl_certfile if tls else None, ssl_keyfile=args.ssl_keyfile if tls else None)
    return 0


if __name__ == "__main__":
    sys.exit(main())

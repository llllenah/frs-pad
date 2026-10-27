"""Облікові записи веб-інтерфейсу з командного рядка (data/users.json).

Пароль вводиться без відображення на екрані і зберігається лише як хеш bcrypt.
Приклади:
    python scripts/manage_users.py add admin@example.com --role admin
    python scripts/manage_users.py list
    python scripts/manage_users.py role operator@example.com admin
    python scripts/manage_users.py block operator@example.com
    python scripts/manage_users.py unblock operator@example.com
    python scripts/manage_users.py passwd operator@example.com
Сервер перечитує users.json лише під час запуску; поки він працює, облікові
записи змінюються на вкладці «Користувачі» веб-інтерфейсу.
"""

from __future__ import annotations

import argparse
import getpass
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pad.auth import ROLE_TEXT, ROLES, Auth, AuthError, UserStore  # noqa: E402
from pad.cli import setup_console  # noqa: E402


def ask_password() -> str:
    first = getpass.getpass("Пароль (не менше 8 символів): ")
    if getpass.getpass("Повторіть пароль: ") != first:
        sys.exit("паролі не збігаються")
    return first


def main() -> int:
    setup_console()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=os.environ.get("PAD_DATA_DIR", str(ROOT / "data")))
    sub = ap.add_subparsers(dest="cmd", required=True)
    add = sub.add_parser("add", help="новий обліковий запис")
    add.add_argument("email")
    add.add_argument("--role", choices=ROLES, default="operator")
    sub.add_parser("list", help="усі облікові записи")
    role = sub.add_parser("role", help="змінити роль")
    role.add_argument("email")
    role.add_argument("role", choices=ROLES)
    for name in ("block", "unblock", "passwd"):
        sub.add_parser(name).add_argument("email")
    args = ap.parse_args()

    data = Path(args.data)
    store = UserStore(data / "users.json")
    auth = Auth(store, None, data / "audit.jsonl")   # лише журнал аудиту, токени не видаються
    try:
        if args.cmd == "list":
            for u in store.list():
                state = "активний" if u.active else "заблоковано"
                print(f"{u.email:40} {ROLE_TEXT[u.role]:15} {state:12} {u.created}")
            if not len(store):
                print("облікових записів немає")
            return 0
        if args.cmd == "add":
            user = store.create(args.email, ask_password(), args.role)
            auth.audit(None, "user.create", user.email, {"role": user.role, "via": "cli"})
            print(f"створено {user.email} ({ROLE_TEXT[user.role]})")
            return 0
        if args.cmd == "role":
            user, changes = store.update(args.email, role=args.role)
        elif args.cmd == "passwd":
            user, changes = store.update(args.email, password=ask_password())
        else:
            user, changes = store.update(args.email, active=args.cmd == "unblock")
        if changes:
            auth.audit(None, "user.update", user.email, {**changes, "via": "cli"})
        print(f"{user.email}: {', '.join(changes) or 'без змін'}")
    except AuthError as exc:
        sys.exit(str(exc))
    return 0


if __name__ == "__main__":
    sys.exit(main())

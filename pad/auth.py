"""Облікові записи, вхід до системи і журнал аудиту (ТЗ, п. 4.1.2.2.4, 4.2.1, 4.2.4).

Паролі зберігаються лише як хеші bcrypt (cost 12). Після входу сервер видає
JWT (HS256) з терміном дії 24 години; токен передається в cookie HttpOnly або в
заголовку Authorization: Bearer. Ролі: «admin» – усі дії, «operator» – моніторинг,
верифікація, журнал і перегляд сеансів без зміни бази осіб і налаштувань.
Після 5 невдалих спроб входу за 10 хвилин обліковий запис блокується на 15 хвилин.
Облікові записи – у файлі users.json, журнал аудиту – audit.jsonl (одна подія на рядок).
"""

from __future__ import annotations

import json
import logging
import os
import re
import secrets
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import bcrypt
import jwt

log = logging.getLogger(__name__)

ROLES = ("admin", "operator")
ROLE_TEXT = {"admin": "адміністратор", "operator": "оператор"}
TOKEN_TTL_S = 24 * 3600
BCRYPT_COST = 12
MAX_FAILURES = 5            # невдалих спроб входу …
FAILURE_WINDOW_S = 600      # … за 10 хвилин
LOCKOUT_S = 900             # блокування на 15 хвилин (FR-04)
MIN_PASSWORD = 8
MIN_SECRET = 32             # байт ключа HS256 (RFC 7518, п. 3.2)
EMAIL_RE = re.compile(r"^[^@\s]{1,64}@[^@\s]+\.[^@\s]{2,}$")
ALGORITHM = "HS256"


class AuthError(Exception):
    """Помилка входу або керування обліковими записами; status – код HTTP."""

    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


@dataclass
class User:
    email: str
    role: str
    password_hash: str
    active: bool = True
    created: str = ""

    def public(self) -> dict:
        """Дані для інтерфейсу – без хешу пароля."""
        return {"email": self.email, "role": self.role, "role_text": ROLE_TEXT[self.role],
                "active": self.active, "created": self.created}


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def normalize_email(email: str) -> str:
    email = (email or "").strip().lower()
    if not EMAIL_RE.match(email) or len(email) > 254:
        raise AuthError("некоректна адреса електронної пошти", 422)
    return email


def check_password(password: str) -> str:
    """Вимоги FR-01: не менше 8 символів, хоча б одна велика літера і одна цифра."""
    if not isinstance(password, str) or len(password) < MIN_PASSWORD:
        raise AuthError(f"пароль має містити не менше {MIN_PASSWORD} символів", 422)
    if not any(c.isupper() for c in password) or not any(c.isdigit() for c in password):
        raise AuthError("пароль має містити хоча б одну велику літеру і одну цифру", 422)
    if len(password.encode("utf-8")) > 72:   # bcrypt враховує лише перші 72 байти
        raise AuthError("пароль задовгий (до 72 байт)", 422)
    return password


class UserStore:
    """Облікові записи у JSON-файлі; кожна зміна записується атомарно (тимчасовий файл і заміна)."""

    def __init__(self, path: str | Path, cost: int = BCRYPT_COST):
        self.path = Path(path)
        self.cost = cost
        self.lock = threading.Lock()
        self.users: dict[str, User] = {}
        if self.path.exists():
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            self.users = {u["email"]: User(**u) for u in raw.get("users", [])}
        # хеш для невідомої адреси: перевірка триває стільки ж, скільки для наявної
        self._dummy = bcrypt.hashpw(b"dummy-password", bcrypt.gensalt(self.cost))

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        data = {"users": [asdict(u) for u in self.users.values()]}
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
        os.replace(tmp, self.path)

    def hash(self, password: str) -> str:
        return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt(self.cost)).decode("ascii")

    def get(self, email: str) -> User | None:
        with self.lock:
            return self.users.get((email or "").strip().lower())

    def list(self) -> list[User]:
        with self.lock:
            return sorted(self.users.values(), key=lambda u: (u.role != "admin", u.email))

    def __len__(self) -> int:
        return len(self.users)

    def create(self, email: str, password: str, role: str) -> User:
        email = normalize_email(email)
        if role not in ROLES:
            raise AuthError("невідома роль", 422)
        pw_hash = self.hash(check_password(password))
        with self.lock:
            if email in self.users:
                raise AuthError("обліковий запис з такою адресою вже існує", 409)
            user = User(email, role, pw_hash, True, now_iso())
            self.users[email] = user
            self._save()
        return user

    def update(self, email: str, role: str | None = None, active: bool | None = None,
               password: str | None = None) -> tuple[User, dict]:
        """Змінює роль, стан або пароль; повертає запис і змінені поля (для журналу аудиту)."""
        if role is not None and role not in ROLES:
            raise AuthError("невідома роль", 422)
        pw_hash = self.hash(check_password(password)) if password is not None else None
        with self.lock:
            user = self.users.get((email or "").strip().lower())
            if user is None:
                raise AuthError("обліковий запис не знайдено", 404)
            changes = {}
            if role is not None and role != user.role:
                changes["role"] = [user.role, role]
            if active is not None and active != user.active:
                changes["active"] = [user.active, active]
            # у системі завжди лишається хоча б один активний адміністратор
            if user.role == "admin" and user.active and (changes.get("role") or changes.get("active")):
                admins = [u for u in self.users.values() if u.role == "admin" and u.active]
                if len(admins) == 1:
                    raise AuthError("це останній активний адміністратор: спершу призначте іншого", 409)
            if "role" in changes:
                user.role = role
            if "active" in changes:
                user.active = active
            if pw_hash is not None:
                user.password_hash = pw_hash
                changes["password"] = "змінено"
            if changes:
                self._save()
            return user, changes

    def verify(self, email: str, password: str) -> User | None:
        user = self.get(email)
        stored = user.password_hash.encode("ascii") if user else self._dummy
        ok = bcrypt.checkpw((password or "").encode("utf-8")[:72], stored)
        return user if ok and user is not None else None


class Auth:
    """Вхід, видача і перевірка JWT, обмеження невдалих спроб, вихід і журнал аудиту."""

    def __init__(self, store: UserStore, secret: str | None = None, audit_path: str | Path | None = None,
                 ttl_s: int = TOKEN_TTL_S, clock=time.time):
        if secret and len(secret.encode("utf-8")) < MIN_SECRET:
            raise AuthError(f"ключ підпису токенів закороткий: потрібно не менше {MIN_SECRET} байт", 422)
        self.store = store
        self.ephemeral = not secret
        # без PAD_JWT_SECRET ключ підпису випадковий: після перезапуску потрібно увійти знову
        self.secret = secret or secrets.token_urlsafe(48)
        self.ttl_s = ttl_s
        self.clock = clock
        self.audit_path = Path(audit_path) if audit_path else None
        self.lock = threading.Lock()
        self.failures: dict[str, deque] = {}
        self.locked_until: dict[str, float] = {}
        self.revoked: dict[str, float] = {}     # jti -> exp, щоб вихід діяв до кінця строку токена

    # ----- вхід і вихід -----

    def login(self, email: str, password: str) -> tuple[str, User, float]:
        key = (email or "").strip().lower()
        now = self.clock()
        with self.lock:
            until = self.locked_until.get(key, 0.0)
            if until > now:
                minutes = max(1, round((until - now) / 60))
                raise AuthError(f"забагато невдалих спроб входу: обліковий запис заблоковано ще на {minutes} хв", 429)
        user = self.store.verify(key, password)
        if user is None or not user.active:
            with self.lock:
                q = self.failures.setdefault(key, deque())
                q.append(now)
                while q and now - q[0] > FAILURE_WINDOW_S:
                    q.popleft()
                if len(q) >= MAX_FAILURES:
                    self.locked_until[key] = now + LOCKOUT_S
                    q.clear()
                    log.warning("вхід %s заблоковано на %d хв після %d невдалих спроб",
                                key, LOCKOUT_S // 60, MAX_FAILURES)
            if user is not None and not user.active:
                raise AuthError("обліковий запис заблоковано адміністратором", 403)
            raise AuthError("неправильна адреса або пароль", 401)
        with self.lock:
            self.failures.pop(key, None)
            self.locked_until.pop(key, None)
        token, exp = self.issue(user)
        log.info("вхід %s (%s)", user.email, user.role)
        return token, user, exp

    def issue(self, user: User) -> tuple[str, float]:
        now = self.clock()
        exp = now + self.ttl_s
        payload = {"sub": user.email, "role": user.role, "iat": int(now), "exp": int(exp),
                   "jti": secrets.token_urlsafe(12)}
        return jwt.encode(payload, self.secret, algorithm=ALGORITHM), exp

    def decode(self, token: str | None) -> tuple[User, dict]:
        """Користувач за токеном; роль і стан беруться з бази, тож блокування діє одразу."""
        if not token:
            raise AuthError("потрібно увійти до системи", 401)
        # строк дії перевіряється нижче за годинником self.clock (у тестах – керованим)
        options = {"require": ["exp", "sub", "jti"], "verify_exp": False, "verify_iat": False}
        try:
            claims = jwt.decode(token, self.secret, algorithms=[ALGORITHM], options=options)
        except jwt.InvalidTokenError as exc:
            raise AuthError("недійсний токен: увійдіть знову", 401) from exc
        if claims["exp"] <= self.clock():
            raise AuthError("сесія завершилася: увійдіть знову", 401)
        with self.lock:
            if claims["jti"] in self.revoked:
                raise AuthError("сесію завершено: увійдіть знову", 401)
        user = self.store.get(claims["sub"])
        if user is None or not user.active:
            raise AuthError("обліковий запис заблоковано або видалено", 401)
        return user, claims

    def logout(self, token: str | None) -> None:
        try:
            user, claims = self.decode(token)
        except AuthError:
            return
        now = self.clock()
        with self.lock:
            self.revoked = {j: e for j, e in self.revoked.items() if e > now}
            self.revoked[claims["jti"]] = claims["exp"]
        log.info("вихід %s", user.email)

    # ----- журнал аудиту -----

    def audit(self, user: User | None, action: str, target: str = "", changes: dict | None = None) -> None:
        """Подія зміни даних: час, користувач, дія, об'єкт і змінені значення."""
        entry = {"time": now_iso(), "user": user.email if user else "", "action": action,
                 "target": target, "changes": changes or {}}
        log.info("аудит: %s %s %s", entry["user"], action, target)
        if self.audit_path is None:
            return
        self.audit_path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock, self.audit_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def audit_tail(self, limit: int = 200) -> list[dict]:
        if self.audit_path is None or not self.audit_path.exists():
            return []
        with self.lock:
            lines = self.audit_path.read_text(encoding="utf-8").splitlines()[-limit:]
        return [json.loads(x) for x in reversed(lines) if x.strip()]


def utc_expiry(exp: float) -> str:
    return datetime.fromtimestamp(exp, timezone.utc).isoformat(timespec="seconds")

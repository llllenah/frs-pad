"""Вхід до системи, JWT, ролі, обмеження спроб входу і журнал аудиту (pad/auth.py, ТЗ п. 4.2.1).

У тестах bcrypt працює з cost 4, щоб набір тестів не сповільнювався; окремий тест
перевіряє, що типове значення – 12.
"""

import json
import time

import bcrypt
import jwt
import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from pad.auth import (BCRYPT_COST, FAILURE_WINDOW_S, LOCKOUT_S, MAX_FAILURES, TOKEN_TTL_S, Auth, AuthError,
                      UserStore)
from pad.server import TOKEN_COOKIE, PadService, create_app
from pad.session import SessionManager
from pad.verifier import ChallengeVerifier
from test_server import JPEG, FakePipeline, frame_message
from test_session import Actor, ConstClassifier

ADMIN, OPERATOR, PASSWORD = "admin@example.com", "operator@example.com", "Correct-horse-1"
SECRET = "test-secret-" + "0" * 32


class Clock:
    def __init__(self, t=None):
        self.t = time.time() if t is None else t

    def __call__(self):
        return self.t


@pytest.fixture
def clock():
    return Clock()


@pytest.fixture
def auth(tmp_path, clock):
    store = UserStore(tmp_path / "users.json", cost=4)
    store.create(ADMIN, PASSWORD, "admin")
    store.create(OPERATOR, PASSWORD, "operator")
    return Auth(store, SECRET, tmp_path / "audit.jsonl", clock=clock)


def make_app(tmp_path, auth):
    manager = SessionManager(ConstClassifier(0.1), verifier=ChallengeVerifier())
    pipeline = FakePipeline(Actor())
    service = PadService(pipeline, manager, tmp_path / "data", None)
    pipeline.service = service
    return TestClient(create_app(service, auth=auth)), service


def login(client, email=ADMIN, password=PASSWORD, remember=False):
    return client.post("/api/auth/login", json={"email": email, "password": password, "remember": remember})


# ----- облікові записи -----

def test_password_is_stored_as_bcrypt_hash(tmp_path):
    store = UserStore(tmp_path / "users.json")
    assert store.cost == BCRYPT_COST == 12
    store.create("a@example.com", PASSWORD, "admin")
    raw = (tmp_path / "users.json").read_text(encoding="utf-8")
    assert PASSWORD not in raw
    stored = json.loads(raw)["users"][0]["password_hash"]
    assert stored.startswith("$2b$12$")
    assert bcrypt.checkpw(PASSWORD.encode(), stored.encode())


def test_store_validation(tmp_path):
    store = UserStore(tmp_path / "users.json", cost=4)
    with pytest.raises(AuthError):
        store.create("not-an-email", PASSWORD, "admin")
    with pytest.raises(AuthError):
        store.create("a@example.com", "short", "admin")
    with pytest.raises(AuthError):
        store.create("a@example.com", PASSWORD, "root")
    store.create("A@Example.com ", PASSWORD, "admin")
    assert store.get("a@example.com").email == "a@example.com"
    with pytest.raises(AuthError) as exc:
        store.create("a@example.com", PASSWORD, "operator")
    assert exc.value.status == 409


def test_users_survive_restart(tmp_path, auth):
    again = UserStore(tmp_path / "users.json", cost=4)
    assert {u.email for u in again.list()} == {ADMIN, OPERATOR}
    assert again.verify(OPERATOR, PASSWORD) is not None


def test_last_active_admin_cannot_be_blocked_or_demoted(auth):
    with pytest.raises(AuthError):
        auth.store.update(ADMIN, active=False)
    with pytest.raises(AuthError):
        auth.store.update(ADMIN, role="operator")
    auth.store.update(OPERATOR, role="admin")
    _, changes = auth.store.update(ADMIN, active=False)
    assert changes == {"active": [True, False]}


# ----- токени -----

def test_token_lifetime_is_24_hours(auth, clock):
    token, _, exp = auth.login(ADMIN, PASSWORD)
    claims = jwt.decode(token, SECRET, algorithms=["HS256"])
    assert claims["exp"] - claims["iat"] == TOKEN_TTL_S == 24 * 3600
    assert claims["sub"] == ADMIN and claims["role"] == "admin"
    clock.t += TOKEN_TTL_S - 1
    assert auth.decode(token)[0].email == ADMIN
    clock.t += 2
    with pytest.raises(AuthError):
        auth.decode(token)


def test_forged_and_foreign_tokens_are_rejected(auth, clock):
    forged = jwt.encode({"sub": ADMIN, "role": "admin", "exp": int(clock.t) + 60, "jti": "x"}, "x" * 40,
                        algorithm="HS256")
    with pytest.raises(AuthError):
        auth.decode(forged)
    unsigned = jwt.encode({"sub": ADMIN, "exp": int(clock.t) + 60, "jti": "x"}, None, algorithm="none")
    with pytest.raises(AuthError):
        auth.decode(unsigned)


def test_role_comes_from_store_not_from_token(auth):
    token, _, _ = auth.login(OPERATOR, PASSWORD)
    auth.store.update(OPERATOR, role="admin")
    assert auth.decode(token)[0].role == "admin"
    auth.store.update(OPERATOR, active=False)
    with pytest.raises(AuthError):
        auth.decode(token)


# ----- обмеження спроб входу -----

def test_lockout_after_five_failures(auth, clock):
    for _ in range(MAX_FAILURES):
        with pytest.raises(AuthError) as exc:
            auth.login(ADMIN, "wrong-password")
        assert exc.value.status == 401
    with pytest.raises(AuthError) as exc:
        auth.login(ADMIN, PASSWORD)            # навіть правильний пароль
    assert exc.value.status == 429
    clock.t += LOCKOUT_S + 1
    assert auth.login(ADMIN, PASSWORD)[1].email == ADMIN


def test_failures_outside_window_do_not_lock(auth, clock):
    for _ in range(MAX_FAILURES - 1):
        with pytest.raises(AuthError):
            auth.login(ADMIN, "wrong-password")
    clock.t += FAILURE_WINDOW_S + 1
    with pytest.raises(AuthError) as exc:
        auth.login(ADMIN, "wrong-password")
    assert exc.value.status == 401
    assert auth.login(ADMIN, PASSWORD)[1].email == ADMIN


def test_unknown_email_gives_same_error(auth):
    with pytest.raises(AuthError) as a:
        auth.login("nobody@example.com", PASSWORD)
    with pytest.raises(AuthError) as b:
        auth.login(ADMIN, "wrong-password")
    assert (a.value.status, str(a.value)) == (b.value.status, str(b.value))


# ----- REST API -----

def test_api_requires_login(tmp_path, auth):
    client, _ = make_app(tmp_path, auth)
    assert client.get("/").status_code == 200                   # сторінка з формою входу
    assert client.get("/static/app.js").status_code == 200
    for path in ("/api/config", "/api/persons", "/api/incidents", "/api/auth/me", "/api/users"):
        assert client.get(path).status_code == 401, path


def test_login_sets_httponly_cookie(tmp_path, auth):
    client, _ = make_app(tmp_path, auth)
    r = login(client)
    assert r.status_code == 200
    assert r.json()["user"] == {"email": ADMIN, "role": "admin", "role_text": "адміністратор", "active": True,
                                "created": r.json()["user"]["created"]}
    cookie = r.headers["set-cookie"]
    assert TOKEN_COOKIE in cookie and "HttpOnly" in cookie and "SameSite=strict" in cookie
    assert "Max-Age" not in cookie                              # без «запам'ятати» – до закриття браузера
    assert client.get("/api/auth/me").json()["user"]["email"] == ADMIN
    assert client.get("/api/config").status_code == 200
    assert f"Max-Age={TOKEN_TTL_S}" in login(client, remember=True).headers["set-cookie"]


def test_bearer_token_for_api_clients(tmp_path, auth):
    client, _ = make_app(tmp_path, auth)
    token = login(client).json()["token"]
    client.cookies.clear()
    assert client.get("/api/persons").status_code == 401
    assert client.get("/api/persons", headers={"Authorization": f"Bearer {token}"}).status_code == 200


def test_wrong_password_is_rejected(tmp_path, auth):
    client, _ = make_app(tmp_path, auth)
    r = login(client, password="wrong-password")
    assert r.status_code == 401
    assert "set-cookie" not in r.headers


def test_logout_revokes_token(tmp_path, auth):
    client, _ = make_app(tmp_path, auth)
    token = login(client).json()["token"]
    assert client.post("/api/auth/logout").status_code == 200
    assert client.get("/api/config", headers={"Authorization": f"Bearer {token}"}).status_code == 401


def test_operator_cannot_change_data(tmp_path, auth):
    client, service = make_app(tmp_path, auth)
    login(client, OPERATOR)
    assert client.get("/api/persons").status_code == 200
    assert client.get("/api/settings").status_code == 200
    denied = [
        client.post("/api/participants/P01/image", content=JPEG, headers={"Content-Type": "image/jpeg"}),
        client.delete("/api/persons/P01"),
        client.put("/api/settings", json={"pad_threshold": 0.9}),
        client.get("/api/users"),
        client.post("/api/users", json={"email": "x@example.com", "password": PASSWORD}),
        client.get("/api/audit"),
        client.get("/api/recordings"),
        client.post("/api/sessions", json={"mode": "record", "claimed_person_id": "P01"}),
    ]
    assert [r.status_code for r in denied] == [403] * len(denied)
    assert service.manager.settings.threshold != 0.9
    assert client.post("/api/sessions", json={"claimed_person_id": "P01"}).status_code != 403


def test_admin_manages_users_and_actions_are_audited(tmp_path, auth):
    client, _ = make_app(tmp_path, auth)
    login(client)
    r = client.post("/api/users", json={"email": "new@example.com", "password": PASSWORD, "role": "operator"})
    assert r.status_code == 200
    assert client.post("/api/users", json={"email": "new@example.com", "password": PASSWORD}).status_code == 409
    assert client.patch("/api/users/new@example.com", json={"role": "admin"}).json()["role"] == "admin"
    assert client.patch("/api/users/new@example.com", json={"active": False}).json()["active"] is False
    assert client.put("/api/settings", json={"pad_threshold": 0.42}).status_code == 200
    assert client.post("/api/participants/P07/image", content=JPEG,
                       headers={"Content-Type": "image/jpeg"}).status_code == 200
    assert client.delete("/api/persons/P07").status_code == 200
    log = client.get("/api/audit").json()
    actions = [e["action"] for e in reversed(log)]
    assert actions == ["user.create", "user.update", "user.update", "settings.update", "person.add", "person.delete"]
    assert all(e["user"] == ADMIN and e["time"] for e in log)
    assert log[2]["changes"]["pad_threshold"][1] == 0.42
    assert not any("password_hash" in json.dumps(e) for e in log)
    users = client.get("/api/users").json()
    assert all("password_hash" not in u for u in users)


def test_blocked_user_is_logged_out_immediately(tmp_path, auth):
    client, _ = make_app(tmp_path, auth)
    token = login(client, OPERATOR).json()["token"]
    auth.store.update(OPERATOR, active=False)
    assert client.get("/api/config", headers={"Authorization": f"Bearer {token}"}).status_code == 401
    assert login(client, OPERATOR).status_code == 403


def test_login_through_api_is_rate_limited(tmp_path, auth):
    client, _ = make_app(tmp_path, auth)
    codes = [login(client, password="wrong-password").status_code for _ in range(MAX_FAILURES)]
    assert codes == [401] * MAX_FAILURES
    assert login(client).status_code == 429


# ----- WebSocket -----

def test_websocket_requires_login(tmp_path, auth):
    client, _ = make_app(tmp_path, auth)
    with pytest.raises(WebSocketDisconnect) as exc:
        with client.websocket_connect("/ws/stream") as ws:
            ws.receive_text()
    assert exc.value.code == 4401


def test_websocket_closes_after_logout(tmp_path, auth):
    client, _ = make_app(tmp_path, auth)
    token = login(client).json()["token"]
    with client.websocket_connect("/ws/stream", headers={"Authorization": f"Bearer {token}"}) as ws:
        ws.send_bytes(frame_message(0.0))
        assert json.loads(ws.receive_text())["type"] == "frame"
        auth.logout(token)
        ws.send_bytes(frame_message(0.1))
        with pytest.raises(WebSocketDisconnect) as exc:
            ws.receive_text()
        assert exc.value.code == 4401


def test_operator_cannot_enroll_over_websocket(tmp_path, auth):
    client, service = make_app(tmp_path, auth)
    login(client, OPERATOR)
    with client.websocket_connect("/ws/stream") as ws:
        ws.send_text(json.dumps({"cmd": "enroll", "person_id": "P09"}))
        reply = json.loads(ws.receive_text())
    assert reply["type"] == "error" and "адміністратору" in reply["message"]
    assert "P09" not in service.memory_enrolled


def test_short_signing_key_is_rejected(tmp_path):
    with pytest.raises(AuthError):
        Auth(UserStore(tmp_path / "users.json", cost=4), "short-secret")


def test_password_needs_uppercase_and_digit(tmp_path):
    store = UserStore(tmp_path / "users.json", cost=4)
    for weak in ("lowercase-only-1", "NoDigitsHere", "Short1"):
        with pytest.raises(AuthError):
            store.create("a@example.com", weak, "operator")


def test_admin_cannot_block_own_account(tmp_path, auth):
    client, _ = make_app(tmp_path, auth)
    login(client)
    auth.store.update(OPERATOR, role="admin")                  # інший адміністратор є
    assert client.patch(f"/api/users/{ADMIN}", json={"active": False}).status_code == 409
    assert auth.store.get(ADMIN).active

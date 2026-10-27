"""Захист сервера: запити зі сторонніх сайтів, адреса сервера, розмір запитів, заголовки."""

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from pad.server import MAX_BODY, create_app, local_hosts, same_origin
from test_server import make_client

JPEG = cv2.imencode(".jpg", np.full((64, 64, 3), 128, np.uint8))[1].tobytes()


def test_same_origin_rules():
    assert same_origin(None, "127.0.0.1:8000")                       # не браузер (curl, тести)
    assert same_origin("http://127.0.0.1:8000", "127.0.0.1:8000")
    assert not same_origin("https://evil.example", "127.0.0.1:8000")
    assert not same_origin("null", "127.0.0.1:8000")                 # sandbox-iframe, file://
    assert not same_origin("http://127.0.0.1:9000", "127.0.0.1:8000")


def test_local_hosts():
    assert local_hosts("127.0.0.1", 8000) == {"127.0.0.1:8000", "localhost:8000", "[::1]:8000"}
    assert local_hosts("192.168.1.5", 8000) == {"192.168.1.5:8000"}


def test_cross_site_enroll_is_rejected(tmp_path):
    client, service = make_client(tmp_path)
    r = client.post("/api/participants/P66/image", content=JPEG,
                    headers={"Origin": "https://evil.example", "Content-Type": "text/plain"})
    assert r.status_code == 403
    assert "P66" not in service.memory_enrolled


def test_cross_site_delete_and_settings_are_rejected(tmp_path):
    client, _ = make_client(tmp_path)
    assert client.delete("/api/persons/P01", headers={"Origin": "https://evil.example"}).status_code == 403
    r = client.put("/api/settings", json={"pad_threshold": 0.99}, headers={"Sec-Fetch-Site": "cross-site"})
    assert r.status_code == 403


def test_same_origin_request_passes(tmp_path):
    client, _ = make_client(tmp_path)
    r = client.put("/api/settings", json={"pad_threshold": 0.4}, headers={"Origin": "http://testserver"})
    assert r.status_code == 200


def test_enroll_requires_image_content_type(tmp_path):
    client, _ = make_client(tmp_path)
    r = client.post("/api/participants/P66/image", content=JPEG, headers={"Content-Type": "text/plain"})
    assert r.status_code == 415


def test_oversized_body_is_rejected(tmp_path):
    client, _ = make_client(tmp_path)
    r = client.post("/api/participants/P66/image", content=b"0" * (MAX_BODY + 1),
                    headers={"Content-Type": "image/jpeg"})
    assert r.status_code == 413


def test_unknown_host_is_rejected(tmp_path):
    _, service = make_client(tmp_path)
    client = TestClient(create_app(service, allowed_hosts={"127.0.0.1:8000"}), base_url="http://127.0.0.1:8000")
    assert client.get("/api/config").status_code == 200
    r = client.get("/api/config", headers={"Host": "attacker.example:8000"})   # DNS rebinding
    assert r.status_code == 400


def test_security_headers(tmp_path):
    client, _ = make_client(tmp_path)
    r = client.get("/")
    assert r.headers["X-Frame-Options"] == "DENY"
    assert "frame-ancestors 'none'" in r.headers["Content-Security-Policy"]
    assert r.headers["X-Content-Type-Options"] == "nosniff"


def test_cross_site_websocket_is_rejected(tmp_path):
    client, _ = make_client(tmp_path)
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect("/ws/stream", headers={"Origin": "https://evil.example"}) as ws:
            ws.receive_text()
    with client.websocket_connect("/ws/stream", headers={"Origin": "http://testserver"}) as ws:
        ws.send_text('{"cmd": "nonsense"}')
        assert ws.receive_text()


def test_path_traversal_in_review_and_persons(tmp_path):
    client, _ = make_client(tmp_path)
    for uid in ("..", "..%2F..%2Fenrolled", "P01%5C..%5C..", "a" * 200):
        assert client.get(f"/api/review/{uid}").status_code in (404, 409, 422)
    assert client.delete("/api/persons/..%2FP01").status_code in (404, 422)


def test_local_hosts_in_container_and_extra_names():
    assert local_hosts("0.0.0.0", 8000) == {"127.0.0.1:8000", "localhost:8000", "[::1]:8000"}
    hosts = local_hosts("0.0.0.0", 8443, ["frs.lan", "Gate.local:443", " "])
    assert {"frs.lan:8443", "gate.local:443", "localhost:8443"} <= hosts

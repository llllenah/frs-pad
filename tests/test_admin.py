"""Тести вкладок «База осіб», «Дашборд» і «Налаштування» (/api/persons, /api/dashboard, /api/settings)."""

import json
import os

import numpy as np
import pytest

from pad.storage import save_embedding
from pad.synthetic import embedding_with_similarity
from test_monitoring import monitor_client, people, send_frames, stranger
from test_server import frame_message, make_client, run_session, strict_json


def test_persons_list_and_delete(tmp_path):
    key = os.urandom(32)
    client, service = make_client(tmp_path, key=key)
    folder = tmp_path / "data" / "enrolled"
    folder.mkdir(parents=True)
    base, _ = people(2)
    save_embedding(folder / "P01.npy", base["P01"], key)
    np.save(folder / "P03.npy", base["P02"])          # застарілий відкритий файл
    service.memory_enrolled["T01"] = base["P02"]       # лише в пам'яті

    data = client.get("/api/persons").json()
    rows = {r["id"]: r for r in data["persons"]}
    assert data["count"] == 3 and data["key_set"] is True
    assert rows["P01"]["stored"] == "encrypted" and rows["P01"]["registered"]
    assert rows["P03"]["stored"] == "plain" and rows["T01"]["stored"] == "memory"

    assert client.delete("/api/persons/P01").json() == {"person_id": "P01", "deleted_files": 1}
    assert not (folder / "P01.npy.enc").exists()
    assert client.delete("/api/persons/P03").json()["deleted_files"] == 1
    assert client.delete("/api/persons/T01").json()["deleted_files"] == 0
    assert client.get("/api/persons").json()["persons"] == []
    assert client.delete("/api/persons/P01").status_code == 404
    assert client.delete("/api/persons/..%2FP01").status_code in (404, 422)
    assert client.delete("/api/persons/P%2001").status_code == 422


def test_deleted_person_is_no_longer_recognized(tmp_path):
    client, service, scene = monitor_client(tmp_path)
    base, rng = people(2)
    for pid, e in base.items():
        service.store_embedding(pid, e)
    reg = {r["id"]: r for r in client.get("/api/persons").json()["persons"]}
    assert reg["P01"]["stored"] == "memory" and reg["P01"]["registered"]
    with client.websocket_connect("/ws/stream") as ws:
        ws.send_text(json.dumps({"cmd": "monitor", "on": True}))
        strict_json(ws.receive_text())
        scene.embeddings = [embedding_with_similarity(base["P01"], 0.9, rng)]
        assert send_frames(ws, 1, 1.0)[0]["monitor"]["faces"][0]["person_id"] == "P01"
        client.delete("/api/persons/P01")
        assert send_frames(ws, 1, 2.0)[0]["monitor"]["faces"][0]["known"] is False


def test_encrypted_person_without_key_is_locked(tmp_path):
    key = os.urandom(32)
    folder = tmp_path / "data" / "enrolled"
    folder.mkdir(parents=True)
    save_embedding(folder / "P01.npy", people(1)[0]["P01"], key)
    client, _ = make_client(tmp_path, key=None)
    assert client.get("/api/persons").json()["persons"][0]["stored"] == "locked"
    assert client.delete("/api/persons/P01").status_code == 200   # видалити файл можна і без ключа
    assert not any(folder.iterdir())


def test_dashboard_counts_only_real_events(tmp_path):
    client, service, scene = monitor_client(tmp_path)
    d = client.get("/api/dashboard").json()
    assert d["counts"] == {"known": 0, "unknown": 0, "pad_bona_fide": 0, "pad_attack": 0, "pad_failed": 0}
    assert d["fps"] is None and d["camera_active"] is False and d["persons"] == 0 and d["incidents"] == []

    base, rng = people(1)
    service.store_embedding("P01", base["P01"])
    with client.websocket_connect("/ws/stream") as ws:
        ws.send_text(json.dumps({"cmd": "monitor", "on": True}))
        strict_json(ws.receive_text())
        scene.embeddings = [embedding_with_similarity(base["P01"], 0.9, rng), stranger(base, rng)]
        send_frames(ws, 4, 1.0)
    sid = client.post("/api/sessions", json={"claimed_person_id": "P01"}).json()["id"]
    run_session(client, sid)
    d = client.get("/api/dashboard").json()
    assert d["counts"]["known"] == 1 and d["counts"]["unknown"] == 1 and d["counts"]["pad_bona_fide"] == 1
    assert d["persons"] == 1 and d["camera_active"] is True and d["fps"] > 0
    assert [e["kind"] for e in d["incidents"]] == ["pad", "unknown"]   # без подій «відома особа»
    # лічильники не залежать від довжини журналу
    for _ in range(service.monitor.max_events + 5):
        service._event("unknown", None, 0.1)
    assert client.get("/api/dashboard").json()["counts"]["unknown"] == service.monitor.max_events + 6


def test_settings_in_memory(tmp_path):
    client, service = make_client(tmp_path, score=0.4)
    s = client.get("/api/settings").json()
    assert s["recognition_threshold"] == 0.5 and s["pad_threshold"] == 0.5
    assert s["classifier_version"] == "const-0.4" and s["challenge_count"] == 3
    assert s["config"] == {"recognition_threshold": 0.5, "pad_threshold": 0.5}
    assert client.put("/api/settings", json={"pad_threshold": 1.5}).status_code == 422
    s = client.put("/api/settings", json={"recognition_threshold": 0.62, "pad_threshold": 0.3}).json()
    assert s["recognition_threshold"] == pytest.approx(0.62) and s["pad_threshold"] == pytest.approx(0.3)
    assert service.gallery.threshold == pytest.approx(0.62)
    # новий поріг атаки діє на наступний сеанс: оцінка 0,4 >= 0,3 – атака
    sid = client.post("/api/sessions", json={"claimed_person_id": "P01"}).json()["id"]
    assert run_session(client, sid)[-1]["session"]["state"] == "attack"
    s = client.put("/api/settings", json={"reset": True}).json()
    assert s["recognition_threshold"] == 0.5 and s["pad_threshold"] == 0.5


def test_frames_mark_camera_activity(tmp_path):
    client, service = make_client(tmp_path)
    with client.websocket_connect("/ws/stream") as ws:
        ws.send_bytes(frame_message(1.0))
        strict_json(ws.receive_text())
    assert client.get("/api/dashboard").json()["camera_active"] is True

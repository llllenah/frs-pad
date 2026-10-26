"""Тести бекенду веб-інтерфейсу (pad/server.py): REST, WebSocket, запис і шифрування.

Камера і моделі не потрібні: FakePipeline замість детекції і MediaPipe повертає
кадри синтетичного користувача (Actor з test_session.py), а браузер імітує
TestClient, який надсилає через WebSocket мітку часу і JPEG.
"""

import json
import os
import struct

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from pad.dataset import build_from_recordings
from pad.detector import FaceBox
from pad.server import ENROLL_FRAMES, PadService, create_app
from pad.session import SessionManager
from pad.storage import decrypt_file, load_embedding, read_json_encrypted
from pad.structures import FrameData
from pad.synthetic import embedding_with_similarity, synthetic_face
from pad.verifier import ChallengeVerifier
from test_session import Actor, ConstClassifier

FPS = 30.0
JPEG = cv2.imencode(".jpg", np.full((72, 128, 3), 128, np.uint8))[1].tobytes()
TIMINGS = {"detect_ms": 1.0, "landmarks_ms": 2.0, "embed_ms": 3.0, "total_ms": 6.0}


class FakePipeline:
    """Конвеєр без камери і моделей: кадр формує синтетичний користувач."""

    def __init__(self, actor: Actor):
        self.actor = actor
        self.service: PadService | None = None
        self.detector = type("Det", (), {"name": "fake-detector"})()
        self.embedder = type("Emb", (), {"name": "fake-embedder"})()
        self.verifier = ChallengeVerifier()
        self.last_box = None

    def _active_session(self):
        for live in reversed(list(self.service.sessions.values())):
            if not live.cancelled and not live.session.state.is_final:
                return live.session
        return None

    def process(self, frame, t, landmarks=True, hands=True):
        session = self._active_session()
        if session is not None:
            data = self.actor.frame(t, session)
        else:
            data = FrameData(t=t, embedding=embedding_with_similarity(self.actor.identity, 0.97, self.actor.rng),
                             det_score=0.99, face=synthetic_face(), size=(1280, 720))
        self.last_box = FaceBox(np.array([500.0, 200.0, 280.0, 320.0]), np.zeros((5, 2)), 0.99)
        return data, dict(TIMINGS)

    def embed_image(self, image):
        return None if image.mean() < 10 else self.actor.enrolled


def make_client(tmp_path, score=0.1, key=None):
    manager = SessionManager(ConstClassifier(score), verifier=ChallengeVerifier())
    pipeline = FakePipeline(Actor())
    service = PadService(pipeline, manager, tmp_path / "data", key)
    pipeline.service = service
    return TestClient(create_app(service)), service


def frame_message(t: float, payload: bytes = JPEG) -> bytes:
    return struct.pack("<d", t) + payload


def strict_json(text: str):
    """JSON без NaN та Infinity: JSON.parse у браузері їх не приймає."""
    def bad(c):
        raise AssertionError(f"недопустиме значення {c} у JSON")
    return json.loads(text, parse_constant=bad)


def run_session(client, session_id: int, max_frames: int = 3000, stop_after: int | None = None):
    """Прив'язує сеанс до WebSocket і надсилає кадри до рішення; повертає всі відповіді."""
    replies = []
    with client.websocket_connect("/ws/stream") as ws:
        ws.send_text(json.dumps({"cmd": "attach", "session_id": session_id}))
        attached = strict_json(ws.receive_text())
        assert attached["type"] == "attached" and attached["session"]["id"] == session_id
        for i in range(max_frames):
            ws.send_bytes(frame_message(100.0 + i / FPS))
            reply = strict_json(ws.receive_text())
            assert reply["type"] == "frame"
            replies.append(reply)
            if stop_after is not None and i + 1 >= stop_after:
                break
            if reply["session"]["state"] in ("bona_fide", "attack", "failed"):
                break
    return replies


def record_body(**kw):
    rec = {"participant": "P01", "consent": True, "consent_ref": "07/2026", "presentation_type": "bona_fide",
           "lighting": "low", "webcam": "Logitech C270", "source": "USB camera",
           "scenario": {"challenge_types": ["cover_left", "cover_right", "cover_mouth", "turn_head"],
                        "challenge_count": 2, "duration_s": 3}}
    rec.update(kw)
    return {"mode": "record", "recording": rec}


# ----- довідкові дані і статика -----

def test_index_and_static_files(tmp_path):
    client, _ = make_client(tmp_path)
    html = client.get("/").text
    assert "Сеанс верифікації" in html and "Запис сеансу для набору даних" in html
    for name in ("app.js", "style.css"):
        assert client.get(f"/static/{name}").status_code == 200


def test_config(tmp_path):
    client, _ = make_client(tmp_path)
    cfg = client.get("/api/config").json()
    assert cfg["settings"]["challenge_count"] == 3 and cfg["settings"]["threshold"] == 0.5
    assert [t["id"] for t in cfg["challenge_types"]] == ["cover_left", "cover_right", "cover_mouth",
                                                          "turn_head", "hand_swipe"]
    assert len(cfg["feature_names"]) == 26
    assert cfg["encryption"] == {"algorithm": "AES-256-GCM", "key_set": False}
    assert cfg["embedder"] == "fake-embedder" and cfg["classifier"] == "const-0.1"


# ----- форма 4.7: сеанс верифікації -----

def test_verify_session_over_websocket(tmp_path):
    client, service = make_client(tmp_path, score=0.1)
    r = client.post("/api/sessions", json={"claimed_person_id": "P01"})
    assert r.status_code == 200
    s = r.json()
    assert s["code"].startswith("VS-") and s["state"] == "created" and s["enrolled"] is None

    replies = run_session(client, s["id"])
    states = [x["session"]["state"] for x in replies]
    assert states[0] == "reference" and "challenge" in states and "features" in states
    during = [x for x in replies if x["session"]["state"] == "challenge"][5]  # t_start – з другого кадру виклику
    assert during["session"]["status"]["remaining"] is not None
    assert during["session"]["measure"] is not None
    assert during["face"] is not None and len(during["face"]) == 468 * 2
    assert during["check"]["face_points"] == 468 and during["box"][4] == pytest.approx(0.99)
    assert isinstance(during["hands"], list) and during["check"]["hands"] == (during["hand"] is not None)

    final = replies[-1]["session"]
    assert final["state"] == "bona_fide"
    assert final["decision"]["decision"] == "bona_fide" and final["decision"]["score"] == pytest.approx(0.1)
    assert "series" not in final["decision"]["explanation"]
    assert final["decision"]["explanation"]["latency"]["after_last_challenge_s"] > 0
    assert [c["state"] for c in final["challenges"]] == ["done"] * 3
    assert len(final["features"]) == 26
    assert len(final["series"]["t"]) == len(final["series"]["s_ref"]) == len(final["series"]["s_enr"])
    assert len(final["spans"]) == 3

    assert client.get(f"/api/sessions/{s['id']}").json()["decision"] == final["decision"]
    stats = client.get("/api/stats").json()
    assert stats["frames"] == len(replies) and len(stats["decisions"]) == 1


def test_attack_decision_and_preliminary_score(tmp_path):
    client, _ = make_client(tmp_path, score=0.9)
    sid = client.post("/api/sessions", json={"claimed_person_id": "P01"}).json()["id"]
    replies = run_session(client, sid)
    prelim = [x["session"]["preliminary"] for x in replies if x["session"]["preliminary"]]
    assert prelim and prelim[0]["n"] == 1 and prelim[0]["count"] == 3
    assert prelim[0]["score"] == pytest.approx(0.9)
    final = replies[-1]["session"]
    assert final["state"] == "attack" and final["decision"]["is_attack"] is True
    assert final["preliminary"] is None


def test_cancel_session(tmp_path):
    client, _ = make_client(tmp_path)
    sid = client.post("/api/sessions", json={"claimed_person_id": "P01"}).json()["id"]
    run_session(client, sid, stop_after=10)  # закриття WebSocket теж перериває сеанс
    assert client.get(f"/api/sessions/{sid}").json()["state"] == "cancelled"
    sid = client.post("/api/sessions", json={"claimed_person_id": "P01"}).json()["id"]
    assert client.post(f"/api/sessions/{sid}/cancel").json()["state"] == "cancelled"
    assert client.post(f"/api/sessions/{sid}/cancel").status_code == 200
    assert client.post("/api/sessions/999/cancel").status_code == 404
    assert client.get("/api/sessions/999").status_code == 404


@pytest.mark.parametrize("person", ["", "../P01", "P 01", "Олена", "x" * 40])
def test_claimed_person_must_be_pseudonym(tmp_path, person):
    client, _ = make_client(tmp_path)
    assert client.post("/api/sessions", json={"claimed_person_id": person}).status_code == 422


# ----- еталони -----

def test_enroll_without_key_stays_in_memory(tmp_path):
    client, service = make_client(tmp_path)
    with client.websocket_connect("/ws/stream") as ws:
        ws.send_text(json.dumps({"cmd": "enroll", "person_id": "P05"}))
        assert strict_json(ws.receive_text())["type"] == "enroll_started"
        for i in range(ENROLL_FRAMES):
            ws.send_bytes(frame_message(i / FPS))
            reply = strict_json(ws.receive_text())
    assert reply["enroll"] == {"person_id": "P05", "progress": 1.0, "done": True, "ok": True, "stored": "memory"}
    assert not (tmp_path / "data" / "enrolled").exists()
    assert {"id": "P05", "stored": "memory"} in client.get("/api/participants").json()
    assert client.post("/api/sessions", json={"claimed_person_id": "P05"}).json()["enrolled"] == "memory"


def test_enroll_with_key_writes_only_encrypted_file(tmp_path):
    key = os.urandom(32)
    client, service = make_client(tmp_path, key=key)
    with client.websocket_connect("/ws/stream") as ws:
        ws.send_text(json.dumps({"cmd": "enroll", "person_id": "P05"}))
        ws.receive_text()
        for i in range(ENROLL_FRAMES):
            ws.send_bytes(frame_message(i / FPS))
            reply = strict_json(ws.receive_text())
    assert reply["enroll"]["stored"] == "encrypted"
    folder = tmp_path / "data" / "enrolled"
    assert [p.name for p in folder.iterdir()] == ["P05.npy.enc"]
    emb = load_embedding(folder / "P05.npy.enc", key)
    assert emb.shape == (128,) and np.linalg.norm(emb) == pytest.approx(1.0, abs=1e-5)
    # після перезапуску сервера еталон читається з файлу
    service.memory_enrolled.clear()
    assert client.get("/api/participants").json() == [{"id": "P05", "stored": "encrypted"}]
    assert client.post("/api/sessions", json={"claimed_person_id": "P05"}).json()["enrolled"] == "encrypted"


def test_encrypted_enrollment_without_key_is_locked(tmp_path):
    key = os.urandom(32)
    client, service = make_client(tmp_path, key=key)
    jpeg = cv2.imencode(".jpg", np.full((64, 64, 3), 200, np.uint8))[1].tobytes()
    assert client.post("/api/participants/T01/image", content=jpeg).json() == {"person_id": "T01",
                                                                               "stored": "encrypted"}
    client2, _ = make_client(tmp_path, key=None)
    assert client2.post("/api/sessions", json={"claimed_person_id": "T01"}).json()["enrolled"] == "locked"


def test_enroll_from_image(tmp_path):
    client, _ = make_client(tmp_path)
    bright = cv2.imencode(".png", np.full((64, 64, 3), 200, np.uint8))[1].tobytes()
    dark = cv2.imencode(".png", np.zeros((64, 64, 3), np.uint8))[1].tobytes()
    assert client.post("/api/participants/T01/image", content=bright).json()["stored"] == "memory"
    r = client.post("/api/participants/T02/image", content=dark)
    assert r.status_code == 422 and "обличчя не виявлено" in r.json()["detail"]
    assert client.post("/api/participants/T03/image", content=b"not an image").status_code == 422
    assert client.post("/api/participants/..%2FT01/image", content=bright).status_code in (404, 422)


# ----- форма 4.8: запис сеансів -----

@pytest.mark.parametrize("change, status, text", [
    ({"consent": False}, 422, "згоди"),
    ({"presentation_type": "attack"}, 422, "інструмент"),
    ({"participant": "../P01"}, 422, "псевдонім"),
    ({"participant": "Олена Коваль"}, 422, "псевдонім"),
    ({"target_face": "T 01", "presentation_type": "attack", "swap_tool": "deep_live_cam"}, 422, "цільове"),
    ({"scenario": {"challenge_types": ["cover_left", "cover_right", "turn_head"]}}, 422, "4 типів"),
    ({"scenario": {"challenge_count": 7}}, 422, "2–5"),
    ({"scenario": {"duration_s": 10}}, 422, "3–5"),
])
def test_record_validation(tmp_path, change, status, text):
    client, _ = make_client(tmp_path, key=os.urandom(32))
    r = client.post("/api/sessions", json=record_body(**change))
    assert r.status_code == status and text in r.json()["detail"]
    assert not (tmp_path / "data" / "recordings").exists() or not any((tmp_path / "data" / "recordings").iterdir())


def test_record_requires_key(tmp_path):
    client, _ = make_client(tmp_path, key=None)
    r = client.post("/api/sessions", json=record_body())
    assert r.status_code == 409 and "PAD_DATASET_KEY" in r.json()["detail"]
    assert client.get("/api/recordings").status_code == 409


def test_record_session_is_saved_encrypted_and_listed(tmp_path):
    key = os.urandom(32)
    client, _ = make_client(tmp_path, score=0.1, key=key)
    s = client.post("/api/sessions", json=record_body()).json()
    assert s["mode"] == "record" and s["claimed_person_id"] == "P01"
    replies = run_session(client, s["id"])
    final = replies[-1]["session"]
    assert final["state"] == "bona_fide" and len(final["challenges"]) == 2

    folder = tmp_path / "data" / "recordings" / final["record_name"]
    assert sorted(p.name for p in folder.iterdir()) == ["session.json.enc", "video.mp4.enc"]
    rec = read_json_encrypted(folder / "session.json.enc", key)
    meta = rec["metadata"]
    assert meta["participant"] == "P01" and meta["consent"] is True and meta["consent_ref"] == "07/2026"
    assert meta["presentation_type"] == "bona_fide" and meta["swap_tool"] == "" and meta["interface"] == "web"
    assert meta["conditions"] == {"lighting": "low", "webcam": "Logitech C270", "source": "USB camera",
                                  "resolution": "128x72"}
    assert meta["scenario"]["challenge_count"] == 2 and meta["scenario"]["duration_s"] == 3
    assert rec["settings"]["challenge_count"] == 2 and rec["settings"]["duration_s"] == 3
    assert rec["annotation"] == {"reviewed": False} and rec["embedder"] == "fake-embedder"
    assert {c["challenge_type"] for c in rec["decision"]["explanation"]["attempts"][-1]} <= set(
        meta["scenario"]["challenge_types"])

    video = decrypt_file(folder / "video.mp4.enc", tmp_path / "video.mp4", key)
    cap = cv2.VideoCapture(str(video))
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == len(rec["decision"]["explanation"]["series"]["t"])
    assert cap.get(cv2.CAP_PROP_FPS) == pytest.approx(FPS, rel=0.05)
    cap.release()

    listing = client.get("/api/recordings").json()
    row = listing["rows"][0]
    assert row["name"] == final["record_name"] and row["id"] == "S-0001" and row["status"] == "pending"
    assert row["participant"] == "P01" and len(row["challenges"]) == 2 and row["resolution"] == "128x72"
    assert listing["summary"] == {"participants": 1, "sessions": 1, "bona_fide": 1, "attack": 0,
                                  "tools": 0, "targets": 0, "reviewed": 0}

    r = client.post(f"/api/recordings/{row['name']}/review", json={"reviewed": True, "note": "мітки перевірено"})
    assert r.json() == {"name": row["name"], "status": "reviewed"}
    assert client.get("/api/recordings").json()["rows"][0]["status"] == "reviewed"
    rec = read_json_encrypted(folder / "session.json.enc", key)
    assert rec["annotation"]["reviewed"] is True and rec["annotation"]["note"] == "мітки перевірено"
    assert client.post("/api/recordings/P99_x/review", json={"reviewed": True}).status_code == 404

    # запис з веб-інтерфейсу читає той самий build_dataset.py, що й записи record_session.py
    rows = build_from_recordings(tmp_path / "data" / "recordings", key=key)
    assert len(rows) == 2 and {r["subject"] for r in rows} == {"P01"} and rows[0]["label"] == 0


def test_attack_recording_claims_target_face(tmp_path):
    key = os.urandom(32)
    client, _ = make_client(tmp_path, score=0.9, key=key)
    body = record_body(presentation_type="attack", swap_tool="deep_live_cam", target_face="T01")
    s = client.post("/api/sessions", json=body).json()
    assert s["claimed_person_id"] == "T01"
    run_session(client, s["id"])
    listing = client.get("/api/recordings").json()
    row = listing["rows"][0]
    assert row["presentation_type"] == "attack" and row["swap_tool"] == "deep_live_cam"
    assert row["target_face"] == "T01" and row["claimed_identity"] == "T01"
    assert listing["summary"]["attack"] == 1 and listing["summary"]["tools"] == 1
    assert listing["summary"]["targets"] == 1


def test_interrupted_recording_keeps_only_encrypted_video(tmp_path):
    key = os.urandom(32)
    client, _ = make_client(tmp_path, key=key)
    s = client.post("/api/sessions", json=record_body()).json()
    run_session(client, s["id"], stop_after=40)
    folder = tmp_path / "data" / "recordings" / s["record_name"]
    assert [p.name for p in folder.iterdir()] == ["video.mp4.enc"]
    row = client.get("/api/recordings").json()["rows"][0]
    assert row["status"] == "incomplete"
    assert client.post(f"/api/recordings/{row['name']}/review", json={}).status_code == 409


def test_two_recordings_in_same_second_get_separate_folders(tmp_path):
    client, _ = make_client(tmp_path, key=os.urandom(32))
    a = client.post("/api/sessions", json=record_body()).json()["record_name"]
    b = client.post("/api/sessions", json=record_body()).json()["record_name"]
    assert a != b and (tmp_path / "data" / "recordings" / b).is_dir()


# ----- WebSocket: некоректні повідомлення -----

def test_websocket_rejects_bad_messages(tmp_path):
    client, _ = make_client(tmp_path)
    with client.websocket_connect("/ws/stream") as ws:
        ws.send_bytes(b"1234")
        assert strict_json(ws.receive_text())["type"] == "frame_error"
        ws.send_bytes(frame_message(0.0, b"not a jpeg"))
        assert strict_json(ws.receive_text())["type"] == "frame_error"
        ws.send_text("{bad json")
        assert strict_json(ws.receive_text())["type"] == "error"
        ws.send_text(json.dumps({"cmd": "attach", "session_id": 42}))
        assert strict_json(ws.receive_text())["message"] == "сеанс не знайдено"
        ws.send_text(json.dumps({"cmd": "enroll", "person_id": "../x"}))
        assert strict_json(ws.receive_text())["type"] == "error"
        ws.send_bytes(frame_message(0.0))
        reply = strict_json(ws.receive_text())
        assert reply["type"] == "frame" and "session" not in reply and reply["fps"] > 0

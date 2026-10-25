"""Тести вкладки «Моніторинг»: ідентифікація 1:N (pad/identify.py) і режим monitor у /ws/stream.

Ембединги синтетичні: вектор з заданою косинусною подібністю до еталона
(embedding_with_similarity), тому відомо, хто має бути розпізнаний.
"""

import json
import math
import os

import numpy as np
import pytest

from pad.detector import FaceBox
from pad.features import l2_normalize
from pad.identify import FaceGallery, MonitorSettings
from pad.runtime import load_config
from pad.storage import save_embedding
from pad.synthetic import embedding_with_similarity
from test_server import TIMINGS, frame_message, make_client, run_session, strict_json


def people(n, seed=0):
    rng = np.random.default_rng(seed)
    return {f"P{i + 1:02d}": l2_normalize(rng.normal(size=128)) for i in range(n)}, rng


def stranger(gallery: dict, rng, max_sim=0.3):
    """Вектор, далекий від усіх еталонів (подібність до кожного < max_sim)."""
    while True:
        v = l2_normalize(rng.normal(size=128))
        if max(float(v @ e) for e in gallery.values()) < max_sim:
            return v


# ----- FaceGallery -----

def test_known_and_unknown_face():
    base, rng = people(3)
    g = FaceGallery(threshold=0.5, use_faiss=False)
    g.replace(base)
    known = embedding_with_similarity(base["P02"], 0.8, rng)
    unknown = stranger(base, rng)
    m_known, m_unknown = g.search(np.stack([known, unknown]))
    assert m_known.known and m_known.person_id == "P02"
    assert m_known.similarity == pytest.approx(0.8, abs=1e-4)
    assert not m_unknown.known and m_unknown.person_id is None
    assert m_unknown.similarity < 0.5 and m_unknown.nearest in base


def test_threshold_decides():
    base, rng = people(2)
    q = embedding_with_similarity(base["P01"], 0.6, rng)
    for thr, expected in [(0.5, "P01"), (0.59, "P01"), (0.61, None), (0.9, None)]:
        g = FaceGallery(threshold=thr, use_faiss=False)
        g.replace(base)
        m = g.search(q)[0]
        assert m.person_id == expected and m.nearest == "P01"


def test_many_enrolled_picks_the_right_person():
    base, rng = people(20, seed=1)
    g = FaceGallery(threshold=0.5, use_faiss=False)
    g.replace(base)
    ids = list(base)
    queries = np.stack([embedding_with_similarity(base[pid], 0.7, rng) for pid in ids])
    assert [m.person_id for m in g.search(queries)] == ids
    # не нормовані запити теж допустимі: подібність косинусна
    assert g.search(3.0 * queries[4])[0].person_id == ids[4]


def test_empty_gallery_add_and_bad_dimension():
    g = FaceGallery(threshold=0.5, use_faiss=False)
    m = g.search(np.ones(128))[0]
    assert not m.known and math.isnan(m.similarity) and m.nearest is None
    assert g.search(np.zeros((0, 128))) == []
    base, rng = people(1)
    g.add("P01", base["P01"])
    assert len(g) == 1 and "P01" in g
    g.add("BAD", np.ones(64))  # інша розмірність – еталон пропускається
    assert g.ids == ["P01"]
    assert not g.search(np.ones(64))[0].known
    g.replace({})
    assert len(g) == 0 and not g.search(base["P01"])[0].known


def test_faiss_gives_the_same_result():
    pytest.importorskip("faiss")
    base, rng = people(10, seed=2)
    q = np.stack([embedding_with_similarity(base[p], s, rng) for p, s in zip(base, np.linspace(0.2, 0.9, 10))])
    a, b = FaceGallery(0.5, use_faiss=False), FaceGallery(0.5, use_faiss=True)
    a.replace(base)
    b.replace(base)
    assert b.backend.startswith("FAISS")
    for x, y in zip(a.search(q), b.search(q)):
        assert x.person_id == y.person_id and x.similarity == pytest.approx(y.similarity, abs=1e-5)


def test_monitor_settings():
    cfg = MonitorSettings.from_dict(load_config().get("monitoring"))
    assert 0.3 < cfg.threshold < 0.9 and cfg.confirm_frames >= 1
    assert MonitorSettings.from_dict(None) == MonitorSettings()
    assert MonitorSettings.from_dict({"threshold": 0.6, "unknown": 1}).threshold == 0.6
    with pytest.raises(ValueError):
        MonitorSettings.from_dict({"threshold": 1.5})
    with pytest.raises(ValueError):
        MonitorSettings.from_dict({"confirm_frames": 0})


# ----- режим monitor у WebSocket -----

class Scene:
    """Обличчя, які «бачить» камера: список ембедингів, рамки ставляться в ряд."""

    def __init__(self):
        self.embeddings: list[np.ndarray] = []

    def faces(self, frame, min_score=0.9, max_faces=5):
        out = [(FaceBox(np.array([100.0 + 300 * i, 200.0, 200.0, 240.0]), np.zeros((5, 2)), 0.99), e)
               for i, e in enumerate(self.embeddings[:max_faces])]
        return out, dict(TIMINGS)


def monitor_client(tmp_path, key=None, **monitoring):
    client, service = make_client(tmp_path, key=key)
    scene = Scene()
    service.pipeline.faces = scene.faces
    service.monitor = MonitorSettings.from_dict({"confirm_frames": 3, **monitoring})
    service.gallery.threshold = service.monitor.threshold
    return client, service, scene


def send_frames(ws, n, t0):
    replies = []
    for i in range(n):
        ws.send_bytes(frame_message(t0 + i / 30))
        r = strict_json(ws.receive_text())
        assert r["type"] == "frame" and "monitor" in r
        replies.append(r)
    return replies


def test_monitor_known_and_unknown_over_websocket(tmp_path):
    client, service, scene = monitor_client(tmp_path)
    base, rng = people(2)
    for pid, e in base.items():
        service.store_embedding(pid, e)  # без ключа – лише в пам'яті сервера
    assert client.get("/api/config").json()["monitoring"]["threshold"] == 0.5

    with client.websocket_connect("/ws/stream") as ws:
        ws.send_text(json.dumps({"cmd": "monitor", "on": True}))
        started = strict_json(ws.receive_text())
        assert started["type"] == "monitor_started"
        assert started["monitor"]["ids"] == ["P01", "P02"] and started["monitor"]["events"] == []

        scene.embeddings = [embedding_with_similarity(base["P01"], 0.85, rng), stranger(base, rng)]
        r = send_frames(ws, 2, 10.0)
        faces = r[-1]["monitor"]["faces"]
        assert [f["person_id"] for f in faces] == ["P01", None]
        assert [f["known"] for f in faces] == [True, False]
        assert faces[0]["similarity"] == pytest.approx(0.85, abs=1e-3) and faces[1]["similarity"] < 0.5
        assert r[-1]["monitor"]["alert"] is None and r[-1]["monitor"]["events"] == []  # ще не підтверджено

        r = send_frames(ws, 5, 11.0)
        mon = r[-1]["monitor"]
        assert mon["alert"]["count"] == 1 and mon["alert"]["similarity"] < 0.5
        assert sorted(e["kind"] for e in mon["events"]) == ["known", "unknown"]  # по одній події, без повторів
        known = next(e for e in mon["events"] if e["kind"] == "known")
        assert known["person_id"] == "P01" and known["similarity"] == pytest.approx(0.85, abs=1e-3)
        assert next(e for e in mon["events"] if e["kind"] == "unknown")["person_id"] is None

        # невідома особа пішла: сповіщення ще трохи тримається, потім зникає
        scene.embeddings = scene.embeddings[:1]
        held = send_frames(ws, 1, 12.0)[0]["monitor"]["alert"]
        assert held["count"] == 1 and held["similarity"] < 0.5
        assert send_frames(ws, 1, 15.0)[0]["monitor"]["alert"] is None

        # обидві особи знову в кадрі після перерви понад event_gap_s – нові події
        scene.embeddings.append(stranger(base, rng))
        mon = send_frames(ws, 3, 40.0)[-1]["monitor"]
        kinds = [e["kind"] for e in mon["events"]]
        assert kinds.count("unknown") == 2 and kinds.count("known") == 2

        ws.send_text(json.dumps({"cmd": "monitor", "on": False}))
        assert strict_json(ws.receive_text())["type"] == "monitor_stopped"
        ws.send_bytes(frame_message(50.0))
        assert "monitor" not in strict_json(ws.receive_text())

    assert len(client.get("/api/monitor").json()["events"]) == 4


def test_monitor_without_faces_and_empty_gallery(tmp_path):
    client, service, scene = monitor_client(tmp_path, confirm_frames=1)
    with client.websocket_connect("/ws/stream") as ws:
        ws.send_text(json.dumps({"cmd": "monitor"}))
        assert strict_json(ws.receive_text())["monitor"]["ids"] == []
        assert send_frames(ws, 1, 1.0)[0]["monitor"]["faces"] == []
        scene.embeddings = [l2_normalize(np.ones(128))]
        mon = send_frames(ws, 1, 2.0)[0]["monitor"]
        assert mon["faces"][0]["known"] is False and mon["faces"][0]["similarity"] is None  # NaN -> null
        assert mon["alert"] is not None and mon["gallery"] == 0


def test_gallery_reads_encrypted_enrolled(tmp_path):
    key = os.urandom(32)
    base, rng = people(3, seed=3)
    folder = tmp_path / "data" / "enrolled"
    folder.mkdir(parents=True)
    for pid, e in base.items():
        save_embedding(folder / f"{pid}.npy", e, key)

    client, service, scene = monitor_client(tmp_path, key=key)
    service.store_embedding("P09", people(1, seed=9)[0]["P01"])  # реєстрація через сервер з ключем – теж у файл
    with client.websocket_connect("/ws/stream") as ws:
        ws.send_text(json.dumps({"cmd": "monitor", "on": True}))
        assert strict_json(ws.receive_text())["monitor"]["ids"] == ["P01", "P02", "P03", "P09"]
        scene.embeddings = [embedding_with_similarity(base["P03"], 0.7, rng)]
        assert send_frames(ws, 1, 1.0)[0]["monitor"]["faces"][0]["person_id"] == "P03"

    # без ключа зашифровані еталони не читаються, про це видно в стані моніторингу
    client2, service2, _ = monitor_client(tmp_path, key=None)
    with client2.websocket_connect("/ws/stream") as ws:
        ws.send_text(json.dumps({"cmd": "monitor", "on": True}))
        mon = strict_json(ws.receive_text())["monitor"]
        assert mon["ids"] == [] and mon["locked"] == ["P01", "P02", "P03", "P09"] and mon["key_set"] is False


def test_enrolled_during_monitoring_is_found(tmp_path):
    client, service, scene = monitor_client(tmp_path)
    with client.websocket_connect("/ws/stream") as ws:
        ws.send_text(json.dumps({"cmd": "monitor", "on": True}))
        strict_json(ws.receive_text())
        base, rng = people(1, seed=4)
        scene.embeddings = [embedding_with_similarity(base["P01"], 0.9, rng)]
        assert send_frames(ws, 1, 1.0)[0]["monitor"]["faces"][0]["known"] is False
        service.store_embedding("P01", base["P01"])  # наприклад, еталон з фото в іншій вкладці
        assert send_frames(ws, 1, 2.0)[0]["monitor"]["faces"][0]["person_id"] == "P01"


def test_monitor_and_verify_session_share_the_stream(tmp_path):
    client, service, scene = monitor_client(tmp_path)
    sid = client.post("/api/sessions", json={"claimed_person_id": "P01", "origin": "monitor"}).json()["id"]
    with client.websocket_connect("/ws/stream") as ws:
        ws.send_text(json.dumps({"cmd": "attach", "session_id": sid}))
        strict_json(ws.receive_text())
        ws.send_text(json.dumps({"cmd": "monitor", "on": True}))
        assert strict_json(ws.receive_text())["type"] == "error"  # під час сеансу – ні
    # сеанс перервано закриттям з'єднання; перевірка з моніторингу до рішення
    sid = client.post("/api/sessions", json={"claimed_person_id": "P01", "origin": "monitor"}).json()["id"]
    replies = run_session(client, sid)
    final = replies[-1]["session"]
    assert final["origin"] == "monitor" and final["state"] == "bona_fide"
    events = client.get("/api/monitor").json()["events"]
    assert events[0]["kind"] == "pad" and events[0]["person_id"] == "P01"
    assert events[0]["decision"] == "bona_fide" and events[0]["code"] == final["code"]
    assert client.post("/api/sessions", json={"claimed_person_id": "P01", "origin": "x"}).status_code == 422

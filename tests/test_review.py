"""Тести журналу інцидентів і перегляду сеансу (pad/review.py, /api/incidents, /api/review).

Записи синтетичні: відео з однотонними кадрами, яскравість кадру кодує його номер,
тому видно, чи сервер віддає саме ті кадри, що відповідають вікну сеансу. Обидва
формати записів створюються в tmp_path і шифруються тимчасовим ключем.
"""

import json
import os

import cv2
import numpy as np
import pytest

from pad.review import MemoryReader, decode_indices, find_sessions
from pad.server import MAX_SNAPSHOTS
from pad.storage import (DecryptionError, encrypt_bytes, read_encrypted_buffer, read_json_encrypted,
                         write_encrypted)
from pad.synthetic import embedding_with_similarity
from test_monitoring import monitor_client, people, send_frames, stranger
from test_server import FPS, frame_message, make_client, strict_json

SIZE = (960, 540)
BIG_JPEG = cv2.imencode(".jpg", np.full((720, 1280, 3), 140, np.uint8))[1].tobytes()


def gray(i: int, step: int, base: int = 5) -> int:
    return base + step * i


def write_video(path, fourcc: str, n: int, step: int, fps: float) -> bytes:
    """Відео з n однотонних кадрів; яскравість кадру i – gray(i, step)."""
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*fourcc), fps, SIZE)
    for i in range(n):
        w.write(np.full((SIZE[1], SIZE[0], 3), gray(i, step), np.uint8))
    w.release()
    return path.read_bytes()


def frame_gray(jpeg: bytes) -> float:
    img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_GRAYSCALE)
    return float(img.mean())


def challenge(kind, t_start, t_end, performed, score=None, **kw):
    return {"challenge_type": kind, "instruction": "…", "performed": performed, "t_start": t_start,
            "t_end": t_end, "performed_at": t_start + 0.5 if performed else None, "score": score,
            "features": None, **kw}


def make_live_recording(root, key, tmp_path, n=30, fps=10.0):
    """Живий запис: video.mp4.enc (mp4v) і session.json.enc, кадр i – series.t[i]."""
    name = "P01_bona_fide_20261026_100000"
    folder = root / name
    folder.mkdir(parents=True)
    data = write_video(tmp_path / "plain_live.mp4", "mp4v", n, 8, fps)
    write_encrypted(folder / "video.mp4", data, key)
    t0 = 100.0
    t = [round(i / fps, 4) for i in range(n)]
    s_ref = [None] * 3 + [0.95] * (n - 3)
    rec = {
        "metadata": {"participant": "P01", "presentation_type": "bona_fide", "claimed_identity": "P01",
                     "conditions": {"lighting": "normal", "resolution": "960x540"}, "interface": "web"},
        "settings": {"before_s": 1.0, "after_s": 1.5, "threshold": 0.5},
        "decision": {"decision": "bona_fide", "score": 0.12, "is_attack": False, "threshold": 0.5,
                     "explanation": {"t0": t0, "duration_s": t[-1], "model_version": "test",
                                     "attempts": [[challenge("cover_left", t0 + 0.5, t0 + 1.5, True, 0.12),
                                                   challenge("turn_head", t0 + 1.6, t0 + 2.4, True, 0.08,
                                                             direction="right")]],
                                     "series": {"t": t, "s_ref": s_ref, "s_enr": s_ref}}},
        "annotation": {"reviewed": False},
    }
    write_encrypted(folder / "session.json", json.dumps(rec).encode("utf-8"), key)
    return name, rec


def make_imported_recording(root, key, tmp_path, with_times=True):
    """Імпортований запис: video.avi.enc (MJPG), frame_times.json.enc, теки сеансів _s01 і _s02."""
    name = "P02_attack_20261026_110000"
    folder = root / name
    folder.mkdir(parents=True)
    fps, n = 10.0, 110
    data = write_video(tmp_path / "plain_imported.avi", "MJPG", n, 2, fps)
    write_encrypted(folder / "video.avi", data, key)
    rng = np.random.default_rng(1)
    start = 3.0 if with_times else 0.0   # без frame_times анотація задана в часі самого файлу
    abs_times = [round(start + i / fps + float(rng.uniform(-0.01, 0.01)), 4) for i in range(n)]
    if with_times:
        write_encrypted(folder / "frame_times.json", json.dumps(abs_times).encode("utf-8"), key)
    write_encrypted(folder / "annotation.json", json.dumps({"format": "frs-pad-remote/1"}).encode("utf-8"), key)
    sessions = []
    for k, (t0, t1) in enumerate([(4.0, 8.0), (7.0, 10.5)], start=1):
        window = [v for v in abs_times if t0 - 1e-6 <= v <= t1 + 1e-6]
        t = [round(v - t0, 4) for v in window]
        s_ref = [0.95 if not (t0 + 1.5 <= v <= t0 + 2.5) else 0.55 for v in window]
        rec = {
            "metadata": {"participant": "P02", "presentation_type": "attack", "swap_tool": "deep_live_cam",
                         "target_face": "P03", "claimed_identity": "P03", "session_index": k,
                         "conditions": {"lighting": "normal", "source": "remote_browser"},
                         "source": "offline_swap", "video_file": "video.avi"},
            "settings": {"before_s": 1.0, "after_s": 1.0, "threshold": 0.5, "recovery_ratio": 0.9},
            "decision": {"decision": "offline", "score": None, "explanation": {
                "claimed_person_id": "P03", "t0": t0, "reference_found": True,
                "attempts": [[challenge("cover_mouth", t0 + 1.5, t0 + 2.5, True),
                              challenge("hand_swipe", t0 + 2.6, t0 + 2.9, False)]],
                "series": {"t": t, "s_ref": s_ref, "s_enr": s_ref}}},
        }
        sub = folder / f"{name}_s{k:02d}"
        sub.mkdir()
        write_encrypted(sub / "session.json", json.dumps(rec).encode("utf-8"), key)
        sessions.append((sub.name, rec, window))
    return name, abs_times, sessions


def rec_end(rec):
    return rec["decision"]["explanation"]["series"]["t"][-1]


def tree(root):
    return sorted(str(p.relative_to(root)) for p in root.rglob("*"))


# ----- розшифрування великих файлів і декодування в пам'яті -----

def test_read_encrypted_buffer_matches_and_checks_tag(tmp_path):
    key = os.urandom(32)
    data = os.urandom(300_000)
    path = tmp_path / "video.avi.enc"
    path.write_bytes(encrypt_bytes(data, "video.avi", key))
    assert bytes(read_encrypted_buffer(path, key, chunk=4096)) == data
    assert bytes(read_encrypted_buffer(path, key)) == data
    with pytest.raises(DecryptionError):
        read_encrypted_buffer(path, os.urandom(32))
    blob = bytearray(path.read_bytes())
    blob[1000] ^= 1
    path.write_bytes(bytes(blob))
    with pytest.raises(DecryptionError):
        read_encrypted_buffer(path, key, chunk=4096)
    renamed = tmp_path / "video.mp4.enc"
    renamed.write_bytes(encrypt_bytes(data, "video.avi", key))
    with pytest.raises(DecryptionError):  # ім'я файлу – associated data
        read_encrypted_buffer(renamed, key)


@pytest.mark.parametrize("fourcc, ext", [("MJPG", "avi"), ("mp4v", "mp4")])
def test_decode_indices_from_memory(tmp_path, fourcc, ext):
    data = bytearray(write_video(tmp_path / f"v.{ext}", fourcc, 40, 6, 20.0))
    frames = decode_indices(data, [12, 13, 20, 39])
    assert len(frames) == 4
    img = cv2.imdecode(np.frombuffer(frames[0], np.uint8), cv2.IMREAD_COLOR)
    assert img.shape[:2] == (360, 640)   # зменшено до 640 пікселів завширшки
    tol = 1.5 if fourcc == "MJPG" else 4.0
    for jpeg, i in zip(frames, [12, 13, 20, 39]):
        assert frame_gray(jpeg) == pytest.approx(gray(i, 6), abs=tol)
    reader = MemoryReader(bytearray(b"abcdef"))
    assert reader.read(2) == b"ab" and reader.seek(-1, 2) == 5 and reader.read() == b"f"


# ----- перегляд записаних сеансів -----

def test_review_live_recording(tmp_path):
    key = os.urandom(32)
    client, service = make_client(tmp_path, key=key)
    root = tmp_path / "data" / "recordings"
    name, rec = make_live_recording(root, key, tmp_path)
    before = tree(tmp_path / "data")

    meta = client.get(f"/api/review/{name}").json()
    assert meta["layout"] == "live" and meta["participant"] == "P01" and meta["presentation_type"] == "bona_fide"
    ser_t = rec["decision"]["explanation"]["series"]["t"]
    assert meta["frames"]["count"] == len(ser_t) and meta["frames"]["t"] == ser_t
    assert (meta["frames"]["width"], meta["frames"]["height"]) == (640, 360)
    assert meta["decision"]["decision"] == "bona_fide" and meta["decision"]["score"] == pytest.approx(0.12)
    assert meta["decision"]["source"] == "session"
    ch = meta["challenges"]
    assert [c["type"] for c in ch] == ["cover_left", "turn_head"]
    assert ch[0]["t_start"] == pytest.approx(0.5) and ch[0]["t_end"] == pytest.approx(1.5)
    assert ch[0]["name"] == "Закрити ліву половину обличчя" and ch[1]["name"] == "Поворот голови праворуч"
    assert ch[1]["performed"] is True and ch[1]["score"] == pytest.approx(0.08)

    for i in (0, 7, len(ser_t) - 1):
        r = client.get(f"/api/review/{name}/frame/{i}.jpg")
        assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg"
        assert r.headers["cache-control"] == "no-store"
        assert frame_gray(r.content) == pytest.approx(gray(i, 8), abs=4.0)
    assert client.get(f"/api/review/{name}/frame/{len(ser_t)}.jpg").status_code == 404
    # розшифровано лише в пам'яті: у data/ нічого не додалося і не змінилося
    assert tree(tmp_path / "data") == before
    assert name in service.review_cache

    row = client.get("/api/recordings").json()["rows"][0]
    assert row["name"] == name and row["reviewable"] is True and row["layout"] == "live"


@pytest.mark.parametrize("with_times", [True, False])
def test_review_imported_recording(tmp_path, with_times):
    key = os.urandom(32)
    client, service = make_client(tmp_path, key=key)   # ConstClassifier – «навчена модель» з оцінкою 0,1
    root = tmp_path / "data" / "recordings"
    name, abs_times, sessions = make_imported_recording(root, key, tmp_path, with_times)
    before = tree(tmp_path / "data")

    for uid, rec, window in sessions:
        meta = strict_json(client.get(f"/api/review/{uid}").text)
        t0 = rec["decision"]["explanation"]["t0"]
        assert meta["layout"] == "imported" and meta["presentation_type"] == "attack"
        assert meta["target_face"] == "P03" and meta["claimed_identity"] == "P03"
        ft = meta["frames"]["t"]
        if with_times:
            assert ft == pytest.approx([v - t0 for v in window], abs=1e-3)
        else:  # без frame_times – мітки часу самого файлу (i / fps), у них і задано t0
            assert ft[0] == pytest.approx(0.0, abs=0.11) and ft[-1] == pytest.approx(rec_end(rec), abs=0.11)
        # кадр k сеансу – саме той кадр відео, мітка якого потрапила у вікно
        for k in (0, len(ft) // 2, len(ft) - 1):
            jpeg = client.get(f"/api/review/{uid}/frame/{k}.jpg").content
            if with_times:
                idx = int(np.argmin([abs(v - (ft[k] + t0)) for v in abs_times]))
            else:
                idx = round((ft[k] + t0) * 10)
            assert frame_gray(jpeg) == pytest.approx(gray(idx, 2), abs=0.9)
        ch = meta["challenges"]
        assert [c["performed"] for c in ch] == [True, False]
        assert ch[0]["t_start"] == pytest.approx(1.5) and ch[0]["name"] == "Закрити рот і підборіддя"
        d = meta["decision"]
        assert d["source"] == "model" and d["decision"] == "bona_fide" and d["score"] == pytest.approx(0.1)
        assert d["note"] == "оцінка поточною моделлю" and ch[0]["model_score"] == pytest.approx(0.1)
        assert ch[1]["model_score"] is None
    assert tree(tmp_path / "data") == before

    listing = client.get("/api/recordings").json()
    rows = {r["name"]: r for r in listing["rows"]}
    assert set(rows) == {uid for uid, _, _ in sessions}
    assert all(r["layout"] == "imported" and r["reviewable"] and r["recording"] == name for r in rows.values())
    assert rows[sessions[0][0]]["performed"] == 1 and rows[sessions[0][0]]["duration_s"] > 3
    assert listing["summary"]["sessions"] == 2 and listing["summary"]["attack"] == 2
    # позначка «розмічено» для сеансу імпортованого запису пишеться в його підтеку
    uid = sessions[0][0]
    assert client.post(f"/api/recordings/{uid}/review", json={"reviewed": True}).status_code == 200
    assert read_json_encrypted(root / name / uid / "session.json.enc", key)["annotation"]["reviewed"] is True


def test_offline_session_without_trained_model(tmp_path):
    key = os.urandom(32)
    client, service = make_client(tmp_path, key=key)
    service.manager.classifier.version = "threshold-default"
    _, _, sessions = make_imported_recording(tmp_path / "data" / "recordings", key, tmp_path)
    d = client.get(f"/api/review/{sessions[0][0]}").json()["decision"]
    assert d["decision"] == "offline" and d["score"] is None and d["source"] == "none"
    assert d["note"] == "оцінка не обчислювалася під час запису"


def test_review_errors(tmp_path):
    key = os.urandom(32)
    root = tmp_path / "data" / "recordings"
    client, service = make_client(tmp_path, key=None)
    name, _ = make_live_recording(root, key, tmp_path)
    r = client.get(f"/api/review/{name}")
    assert r.status_code == 409 and "PAD_DATASET_KEY" in r.json()["detail"]

    client, _ = make_client(tmp_path, key=os.urandom(32))   # інший ключ
    r = client.get(f"/api/review/{name}")
    assert r.status_code == 409 and "не розшифровано" in r.json()["detail"]

    client, _ = make_client(tmp_path, key=key)
    for bad in ("P99_x", "..", "%2E%2E%2Fdata", "P01 x"):
        assert client.get(f"/api/review/{bad}").status_code == 404
    (root / "P05_bona_fide_20261026_120000").mkdir()
    (root / "P05_bona_fide_20261026_120000" / "video.mp4.enc").write_bytes(b"x")
    r = client.get("/api/review/P05_bona_fide_20261026_120000")
    assert r.status_code == 409 and "анотації немає" in r.json()["detail"]
    rows = {x["name"]: x for x in client.get("/api/recordings").json()["rows"]}
    assert rows["P05_bona_fide_20261026_120000"]["status"] == "incomplete"
    assert rows["P05_bona_fide_20261026_120000"]["reviewable"] is False


def test_find_sessions_layouts(tmp_path):
    key = os.urandom(32)
    root = tmp_path / "recordings"
    make_live_recording(root, key, tmp_path)
    make_imported_recording(root, key, tmp_path)
    refs = find_sessions(sorted(d for d in root.iterdir() if d.is_dir()))
    assert [(r.uid, r.layout) for r in refs] == [
        ("P01_bona_fide_20261026_100000", "live"),
        ("P02_attack_20261026_110000_s01", "imported"), ("P02_attack_20261026_110000_s02", "imported")]
    assert refs[1].recording.name == "P02_attack_20261026_110000"


# ----- журнал інцидентів -----

def test_incident_journal_unknown_person_with_snapshot(tmp_path):
    client, service, scene = monitor_client(tmp_path)
    base, rng = people(2)
    for pid, e in base.items():
        service.store_embedding(pid, e)
    with client.websocket_connect("/ws/stream") as ws:
        ws.send_text(json.dumps({"cmd": "monitor", "on": True}))
        strict_json(ws.receive_text())
        scene.embeddings = [embedding_with_similarity(base["P01"], 0.85, rng), stranger(base, rng)]
        for i in range(4):
            ws.send_bytes(frame_message(10.0 + i / 30, BIG_JPEG))
            mon = strict_json(ws.receive_text())["monitor"]
        # ще кадри з тією самою невідомою особою – нова подія не з'являється
        send_frames(ws, 5, 11.0)
    events = client.get("/api/incidents").json()
    assert sorted(e["kind"] for e in events) == ["known", "unknown"]
    unknown = next(e for e in events if e["kind"] == "unknown")
    assert unknown["level"] == "alarm" and unknown["person_id"] is None and unknown["similarity"] < 0.5
    assert unknown["snapshot"] is True and unknown["id"] >= 1 and unknown["time"] and unknown["date"]
    assert next(e for e in events if e["kind"] == "known")["level"] == "info"
    assert [e["id"] for e in mon["events"]] == [e["id"] for e in events]
    r = client.get(f"/api/incidents/{unknown['id']}/snapshot.jpg")
    assert r.status_code == 200 and r.headers["cache-control"] == "no-store"
    img = cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR)
    assert img.shape[0] == 112
    assert client.get("/api/incidents/999/snapshot.jpg").status_code == 404
    assert not (tmp_path / "data").exists()   # знімки лише в пам'яті


def test_snapshots_are_limited(tmp_path):
    _, service = make_client(tmp_path)
    for i in range(MAX_SNAPSHOTS + 5):
        service._event("unknown", None, 0.1, snapshot=b"jpeg")
    assert len(service.snapshots) == MAX_SNAPSHOTS and min(service.snapshots) == 6
    flags = [e["snapshot"] for e in service.incidents()]
    assert flags[:MAX_SNAPSHOTS] == [True] * MAX_SNAPSHOTS and not any(flags[MAX_SNAPSHOTS:])


def run_big_session(client, sid):
    with client.websocket_connect("/ws/stream") as ws:
        ws.send_text(json.dumps({"cmd": "attach", "session_id": sid}))
        strict_json(ws.receive_text())
        for i in range(3000):
            ws.send_bytes(frame_message(100.0 + i / FPS, BIG_JPEG))
            s = strict_json(ws.receive_text())["session"]
            if s["state"] in ("bona_fide", "attack", "failed"):
                return s


@pytest.mark.parametrize("score, decision, level", [(0.1, "bona_fide", "ok"), (0.9, "attack", "alarm")])
def test_verification_result_in_journal_and_review(tmp_path, score, decision, level):
    client, service = make_client(tmp_path, score=score)
    sid = client.post("/api/sessions", json={"claimed_person_id": "P01", "origin": "monitor"}).json()["id"]
    final = run_big_session(client, sid)
    assert final["decision"]["decision"] == decision
    ev = client.get("/api/incidents").json()[0]
    assert ev["kind"] == "pad" and ev["decision"] == decision and ev["level"] == level
    assert ev["code"] == final["code"] and ev["review"] == final["code"] and ev["snapshot"] is True
    assert ev["origin"] == "monitor"
    assert client.get(f"/api/incidents/{ev['id']}/snapshot.jpg").status_code == 200

    meta = strict_json(client.get(f"/api/review/{ev['review']}").text)   # кадри – у пам'яті сервера, ключ не потрібен
    assert meta["layout"] == "memory" and meta["claimed_identity"] == "P01"
    assert meta["decision"]["decision"] == decision and meta["decision"]["score"] == pytest.approx(score)
    assert meta["frames"]["count"] > 20 and meta["frames"]["width"] == 640
    assert len(meta["challenges"]) == 3 and all(c["performed"] for c in meta["challenges"])
    t = meta["frames"]["t"]
    assert t[0] == pytest.approx(0.0) and all(b - a >= 0.049 for a, b in zip(t, t[1:]))
    assert len(meta["series"]["t"]) >= len(t)
    r = client.get(f"/api/review/{ev['review']}/frame/{len(t) - 1}.jpg")
    assert r.status_code == 200 and r.headers["cache-control"] == "no-store"
    assert not (tmp_path / "data").exists()


def test_old_verification_frames_are_dropped(tmp_path):
    from pad.server import MEMORY_REVIEWS
    client, service = make_client(tmp_path)
    codes = []
    for _ in range(MEMORY_REVIEWS + 1):
        sid = client.post("/api/sessions", json={"claimed_person_id": "P01"}).json()["id"]
        codes.append(run_big_session(client, sid)["code"])
    assert list(service.memory_reviews) == codes[1:]
    events = {e["code"]: e for e in client.get("/api/incidents").json()}
    assert events[codes[0]]["review"] is None and events[codes[-1]]["review"] == codes[-1]
    assert client.get(f"/api/review/{codes[0]}").status_code == 409   # без ключа і без кадрів у пам'яті

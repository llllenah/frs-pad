"""Шифрування записів пілотного набору (FR-65): AES-256-GCM, ключ зі змінної середовища."""

import base64
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from pad.app import DemoApp
from pad.cli import load_enrolled
from pad.dataset import build_from_recordings, read_csv, recording_files, rows_from_recording
from pad.session import SessionManager
from pad.settings import PadSettings
from pad.storage import (KEY_ENV, MAGIC, NONCE_SIZE, DatasetKeyError, DecryptionError, decrypt_file,
                         encrypt_file, find_embedding, load_embedding, load_key, load_key_if_set,
                         read_encrypted, read_json_encrypted, save_embedding, write_encrypted)
from test_session import Actor, ConstClassifier, run

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def key():
    return os.urandom(32)


@pytest.fixture
def env_key(monkeypatch, key):
    monkeypatch.setenv(KEY_ENV, base64.b64encode(key).decode())
    return key


def recorded_session(tmp_path, key, participant="P01", kind="attack"):
    """Тека сеансу із зашифрованим session.json.enc, як після record_session.py."""
    s = run(SessionManager(ConstClassifier(0.9 if kind == "attack" else 0.1)), Actor())
    rec = {"metadata": {"participant": participant, "presentation_type": kind,
                        "swap_tool": "deep_live_cam" if kind == "attack" else ""},
           "settings": PadSettings().to_dict(), "decision": json.loads(s.decision.to_json())}
    folder = tmp_path / "recordings" / f"{participant}_{kind}_20261017_120000"
    folder.mkdir(parents=True)
    return write_encrypted(folder / "session.json", json.dumps(rec).encode("utf-8"), key), s


def test_file_round_trip(tmp_path, key):
    data = os.urandom(100_000)
    plain = tmp_path / "video.mp4"
    plain.write_bytes(data)
    enc = encrypt_file(plain, key)
    assert enc.name == "video.mp4.enc" and not plain.exists()
    blob = enc.read_bytes()
    assert blob.startswith(MAGIC) and data not in blob
    assert len(blob) == len(MAGIC) + NONCE_SIZE + len(data) + 16
    assert read_encrypted(enc, key) == data
    out = decrypt_file(enc, tmp_path / "view.mp4", key)
    assert out.read_bytes() == data


def test_nonce_is_random(tmp_path, key):
    a = write_encrypted(tmp_path / "x.bin", b"same", key).read_bytes()
    b = write_encrypted(tmp_path / "x.bin", b"same", key).read_bytes()
    head = len(MAGIC) + NONCE_SIZE
    assert a[len(MAGIC):head] != b[len(MAGIC):head] and a != b


def test_wrong_key_is_rejected(tmp_path, key):
    enc = write_encrypted(tmp_path / "session.json", b'{"a": 1}', key)
    with pytest.raises(DecryptionError, match="неправильний ключ"):
        read_encrypted(enc, os.urandom(32))


def test_renamed_or_modified_file_is_rejected(tmp_path, key):
    enc = write_encrypted(tmp_path / "session.json", b'{"a": 1}', key)
    renamed = enc.with_name("video.mp4.enc")
    renamed.write_bytes(enc.read_bytes())
    with pytest.raises(DecryptionError):
        read_encrypted(renamed, key)               # associated data – ім'я файлу
    blob = bytearray(enc.read_bytes())
    blob[-20] ^= 1
    enc.write_bytes(bytes(blob))
    with pytest.raises(DecryptionError):
        read_encrypted(enc, key)
    plain = tmp_path / "session.json"
    plain.write_text("{}", encoding="utf-8")
    with pytest.raises(DecryptionError, match="не зашифрований"):
        read_encrypted(plain.rename(tmp_path / "other.json.enc"), key)


def test_key_from_environment(monkeypatch, key):
    monkeypatch.delenv(KEY_ENV, raising=False)
    with pytest.raises(DatasetKeyError, match="не задано"):
        load_key()
    monkeypatch.setenv(KEY_ENV, "не base64!")
    with pytest.raises(DatasetKeyError, match="base64"):
        load_key()
    monkeypatch.setenv(KEY_ENV, base64.b64encode(os.urandom(16)).decode())
    with pytest.raises(DatasetKeyError, match="32 байти"):
        load_key()
    monkeypatch.setenv(KEY_ENV, base64.b64encode(key).decode())
    assert load_key() == key


def test_dataset_reads_encrypted_session_json(tmp_path, env_key):
    enc, s = recorded_session(tmp_path, env_key)
    rec = read_json_encrypted(enc)
    assert rec["metadata"]["participant"] == "P01"
    rows = rows_from_recording(enc)
    assert len(rows) == 3 and all(r["label"] == 1 for r in rows)
    assert rows[0]["session_uid"] == enc.parent.name
    for r, ch in zip(rows, s.challenges):
        assert r["s_ref_during_min"] == pytest.approx(float(ch.features.as_dict()["s_ref_during_min"]), abs=1e-3)
    with pytest.raises(DecryptionError):
        rows_from_recording(enc, key=os.urandom(32))


def test_build_from_mixed_recordings(tmp_path, env_key):
    enc, _ = recorded_session(tmp_path, env_key, "P01", "attack")
    plain = enc.parent.parent / "P02_bona_fide_20261017_120500" / "session.json"
    plain.parent.mkdir()
    plain.write_bytes(read_encrypted(recorded_session(tmp_path / "tmp", env_key, "P02", "bona_fide")[0]))
    (enc.parent / "session.json").write_bytes(read_encrypted(enc))   # розшифрована копія поруч
    assert recording_files(tmp_path / "recordings") == [enc, plain]
    rows = build_from_recordings(tmp_path / "recordings")
    assert len(rows) == 6
    assert sorted({r["subject"] for r in rows}) == ["P01", "P02"]


def test_demo_app_stores_only_encrypted_files(tmp_path, key):
    with pytest.raises(ValueError, match="ключ"):
        DemoApp(None, None, None, record_dir=tmp_path)
    s = run(SessionManager(ConstClassifier(0.1)), Actor())
    pipeline = SimpleNamespace(embedder=SimpleNamespace(name="stub"))
    manager = SimpleNamespace(settings=PadSettings())
    app = DemoApp(None, pipeline, manager, record_dir=tmp_path, dataset_key=key,
                  metadata={"participant": "P01", "presentation_type": "bona_fide"})
    app.session = s
    app.rec_path = tmp_path / "P01_bona_fide_20261017_120000"
    app.rec_path.mkdir()
    folder = app.rec_path
    frame = np.full((120, 160, 3), 128, np.uint8)
    app._open_writer(frame, 30.0)
    for _ in range(10):
        app.writer.write(frame)
    app._finish_session()
    assert sorted(p.name for p in folder.iterdir()) == ["session.json.enc", "video.mp4.enc"]
    assert app.writer is None and app.rec_path is None
    rec = read_json_encrypted(folder / "session.json.enc", key)
    assert rec["embedder"] == "stub" and rec["decision"]["decision"] == "bona_fide"
    video = decrypt_file(folder / "video.mp4.enc", tmp_path / "video.mp4", key)
    cap = cv2.VideoCapture(str(video))
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 10
    cap.release()


def test_unfinished_session_video_is_encrypted(tmp_path, key):
    app = DemoApp(None, None, None, record_dir=tmp_path, dataset_key=key)
    app.rec_path = tmp_path / "P01_attack_20261017_120000"
    app.rec_path.mkdir()
    folder = app.rec_path
    (folder / "video.mp4").write_bytes(b"frames")
    app._close_recording()
    assert [p.name for p in folder.iterdir()] == ["video.mp4.enc"]
    assert read_encrypted(folder / "video.mp4.enc", key) == b"frames"


def test_embedding_round_trip(tmp_path, key):
    emb = np.random.default_rng(0).standard_normal(128).astype(np.float32)
    plain = tmp_path / "P01.npy"
    np.save(plain, emb + 1)                        # відкритий файл попередньої реєстрації
    enc = save_embedding(plain, emb, key)
    assert enc.name == "P01.npy.enc" and not plain.exists()
    assert emb.tobytes() not in enc.read_bytes()
    loaded = load_embedding(enc, key)
    assert loaded.dtype == np.float32 and np.array_equal(loaded, emb)
    assert find_embedding(plain) == enc
    assert save_embedding(enc, emb, key) == enc    # шлях з .enc не дає .enc.enc
    with pytest.raises(DecryptionError, match="неправильний ключ"):
        load_embedding(enc, os.urandom(32))
    other = enc.with_name("P02.npy.enc")           # ембединг прив'язано до псевдоніма
    other.write_bytes(enc.read_bytes())
    with pytest.raises(DecryptionError):
        load_embedding(other, key)


def test_plain_embedding_and_missing_key(tmp_path, monkeypatch, key):
    monkeypatch.delenv(KEY_ENV, raising=False)
    assert load_key_if_set() is None
    emb = np.arange(128, dtype=np.float32)
    assert find_embedding(tmp_path / "P01.npy") is None
    np.save(tmp_path / "P01.npy", emb)
    assert find_embedding(tmp_path / "P01.npy.enc") == tmp_path / "P01.npy"
    assert np.array_equal(load_embedding(tmp_path / "P01.npy"), emb)   # відкритий .npy – без ключа
    enc = save_embedding(tmp_path / "P02.npy", emb, key)
    with pytest.raises(DatasetKeyError, match="не задано"):
        load_embedding(enc)
    with pytest.raises(SystemExit, match=KEY_ENV):
        load_enrolled(str(enc), None)
    monkeypatch.setenv(KEY_ENV, "не base64!")
    with pytest.raises(DatasetKeyError, match="base64"):
        load_key_if_set()
    monkeypatch.setenv(KEY_ENV, base64.b64encode(key).decode())
    assert load_key_if_set() == key
    assert np.array_equal(load_embedding(enc), emb)


def test_demo_app_enrollment_is_encrypted(tmp_path, monkeypatch, key):
    emb = np.random.default_rng(1).standard_normal(128).astype(np.float32)
    monkeypatch.setattr("pad.app.enroll_embedding", lambda frames, min_score: emb)
    manager = SimpleNamespace(settings=PadSettings())
    with pytest.raises(ValueError, match="ключ"):
        DemoApp(None, None, manager, enroll_path=tmp_path / "demo.npy")
    for enroll_path, dataset_key in ((None, None), (tmp_path / "enrolled" / "P01.npy", key)):
        app = DemoApp(None, None, manager, enroll_path=enroll_path, dataset_key=dataset_key)
        app.enroll_frames = []
        for _ in range(30):
            app._handle_enrollment(None)
        assert app.enroll_frames is None and np.array_equal(app.enrolled, emb)
    assert [p.name for p in tmp_path.rglob("*") if p.is_file()] == ["P01.npy.enc"]
    assert np.array_equal(load_embedding(tmp_path / "enrolled" / "P01.npy.enc", key), emb)


def script(name, *args, key=None):
    env = {k: v for k, v in os.environ.items() if k != KEY_ENV}
    if key is not None:
        env[KEY_ENV] = base64.b64encode(key).decode()
    return subprocess.run([sys.executable, str(ROOT / "scripts" / name), *map(str, args)], env=env,
                          capture_output=True, text=True, encoding="utf-8", timeout=120)


def test_build_dataset_script_on_encrypted_recordings(tmp_path, key):
    recorded_session(tmp_path, key, "P01", "attack")
    recorded_session(tmp_path, key, "P02", "bona_fide")
    out = tmp_path / "features.csv"
    res = script("build_dataset.py", tmp_path / "recordings", "-o", out, key=key)
    assert res.returncode == 0, res.stderr
    rows = read_csv(out)
    assert len(rows) == 6 and sorted({r["label"] for r in rows}) == [0, 1]
    res = script("build_dataset.py", tmp_path / "recordings", "-o", out)
    assert res.returncode != 0 and KEY_ENV in res.stderr
    res = script("build_dataset.py", tmp_path / "recordings", "-o", out, key=os.urandom(32))
    assert res.returncode != 0 and "неправильний ключ" in res.stderr


def test_decrypt_recording_script(tmp_path, key):
    enc, _ = recorded_session(tmp_path, key)
    (enc.parent / "video.mp4.enc").write_bytes(
        write_encrypted(tmp_path / "video.mp4", b"frames", key).read_bytes())
    res = script("decrypt_recording.py", "--new-key")
    assert res.returncode == 0 and len(base64.b64decode(res.stdout.strip())) == 32
    res = script("decrypt_recording.py", tmp_path / "recordings", key=key)
    assert res.returncode == 0, res.stderr
    assert "учасник P01" in res.stdout and "рішення attack" in res.stdout
    assert not list(enc.parent.glob("*.json")) and not list(enc.parent.glob("*.mp4"))
    res = script("decrypt_recording.py", enc.parent, "-o", tmp_path / "view", key=key)
    assert res.returncode == 0, res.stderr
    view = tmp_path / "view" / enc.parent.name
    assert (view / "video.mp4").read_bytes() == b"frames"
    assert json.loads((view / "session.json").read_text(encoding="utf-8"))["metadata"]["participant"] == "P01"


def test_demo_script_encrypted_enrolled_needs_key(tmp_path, key):
    enc = save_embedding(tmp_path / "P01.npy", np.zeros(128, np.float32), key)
    res = script("demo.py", "--enrolled", enc)
    assert res.returncode != 0 and KEY_ENV in res.stderr
    res = script("demo.py", "--enrolled", enc, key=os.urandom(32))
    assert res.returncode != 0 and "неправильний ключ" in res.stderr

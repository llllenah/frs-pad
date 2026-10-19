"""Віддалений запис: анотація, офлайн-обробка відео, імпорт і підміна обличчя."""

import base64
import copy
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from pad.challenges import HandSwipeChallenge, HeadTurnChallenge, OcclusionChallenge
from pad.dataset import rows_from_recording
from pad.mtcnn import MTCNN
from pad.offline import (AnnotationError, VideoFrames, challenge_from_annotation, face_ratio, session_records,
                         validate_annotation)
from pad.settings import PadSettings
from pad.storage import generate_key, read_json_encrypted
from pad.structures import FrameData

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
LFW = ROOT / "data" / "lfw" / "lfw"
SWAP = MODELS / "swap"

ANN = {
    "format": "frs-pad-remote/1", "participant": "P09", "consent": True,
    "conditions": {"lighting": "day", "source": "remote_browser"},
    "enroll": {"t_start": 0.5, "t_end": 1.5},
    "sessions": [{"reference": [2.0, 3.0],
                  "challenges": [{"challenge_type": "cover_left", "t_start": 3.0, "t_end": 4.0},
                                 {"challenge_type": "turn_head", "direction": "right", "t_start": 4.0, "t_end": 5.0}]}],
}


def unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def test_valid_annotation_passes():
    validate_annotation(ANN)


@pytest.mark.parametrize("patch, message", [
    ({"format": "other"}, "format"),
    ({"consent": False}, "згоду"),
    ({"participant": ""}, "псевдоніма"),
    ({"sessions": []}, "сеансу"),
])
def test_annotation_header_errors(patch, message):
    with pytest.raises(AnnotationError, match=message):
        validate_annotation({**ANN, **patch})


def test_annotation_challenge_errors():
    bad = copy.deepcopy(ANN)
    bad["sessions"][0]["challenges"][1]["direction"] = "up"
    with pytest.raises(AnnotationError, match="direction"):
        validate_annotation(bad)
    bad = copy.deepcopy(ANN)
    bad["sessions"][0]["challenges"][0]["challenge_type"] = "blink"
    with pytest.raises(AnnotationError, match="невідомий"):
        validate_annotation(bad)
    bad = copy.deepcopy(ANN)
    bad["sessions"][0]["challenges"][1]["t_start"] = 3.5   # перекривається з попереднім викликом
    with pytest.raises(AnnotationError, match="впорядковані"):
        validate_annotation(bad)


def test_challenge_from_annotation_types_and_duration():
    s = PadSettings()
    occ = challenge_from_annotation({"challenge_type": "cover_mouth", "t_start": 1.0, "t_end": 4.0}, s)
    turn = challenge_from_annotation({"challenge_type": "turn_head", "direction": "left",
                                      "t_start": 1.0, "t_end": 3.5}, s)
    swipe = challenge_from_annotation({"challenge_type": "hand_swipe", "t_start": 0.0, "t_end": 3.0}, s)
    assert isinstance(occ, OcclusionChallenge) and occ.type == "cover_mouth"
    assert isinstance(turn, HeadTurnChallenge) and turn.type == "turn_head" and turn.direction == "left"
    assert isinstance(swipe, HandSwipeChallenge)
    assert occ.duration_s == pytest.approx(3.0) and turn.duration_s == pytest.approx(2.5)


def synthetic_frames(fps=30.0, t_end=7.0):
    """Обличчя P09 весь час; під час cover_left (3–4 с) ембединг відхиляється, як при перекритті."""
    rng = np.random.default_rng(1)
    base = unit(rng.normal(size=128))
    frames = []
    for i in range(int(t_end * fps)):
        t = i / fps
        if 3.2 <= t <= 3.8:
            emb = unit(base + 1.5 * unit(rng.normal(size=128)))
        else:
            emb = unit(base + 0.05 * rng.normal(size=128))
        missing = 3.4 <= t <= 3.6   # обличчя повністю закрите
        frames.append(FrameData(t=t, embedding=None if missing else emb, det_score=0.0 if missing else 0.99))
    return VideoFrames(frames=frames, fps=fps, size=(1280, 720)), base


def test_face_ratio_ignores_occlusion_windows():
    vf, _ = synthetic_frames()
    assert face_ratio(vf, ANN) == pytest.approx(1.0)
    for f in vf.frames[:21]:          # 0,7 с без обличчя поза викликами
        f.embedding, f.det_score = None, 0.0
    assert face_ratio(vf, ANN) < 0.95


def test_session_records_match_live_format(tmp_path):
    vf, base = synthetic_frames()
    recs = session_records(vf, ANN, PadSettings(), base, {"participant": "P09", "presentation_type": "bona_fide",
                                                         "claimed_identity": "P09"})
    assert len(recs) == 1
    rec = recs[0]
    expl = rec["decision"]["explanation"]
    assert rec["metadata"]["session_index"] == 1
    assert expl["t0"] == 2.0 and expl["reference_found"]
    assert [c["challenge_type"] for c in expl["attempts"][0]] == ["cover_left", "turn_head"]
    t = expl["series"]["t"]
    assert t[0] == pytest.approx(0.0) and t[-1] <= 5.0 + 1.5 - 2.0 + 1e-6
    assert all(a < b for a, b in zip(t, t[1:]))
    s_ref = expl["series"]["s_ref"]
    assert None in s_ref and min(v for v in s_ref if v is not None) < 0.6
    # FrameData без точок Face Mesh і кисті – виклики не виконано; позначаємо вручну, щоб перевірити,
    # що build_dataset читає запис так само, як запис живого сеансу
    for c in expl["attempts"][0]:
        c["performed"] = True
    path = tmp_path / "s" / "session.json"
    path.parent.mkdir()
    path.write_text(json.dumps(rec), encoding="utf-8")
    rows = rows_from_recording(path)
    assert [r["challenge_type"] for r in rows] == ["cover_left", "turn_head"]
    assert rows[0]["label"] == 0 and rows[0]["subject"] == "P09"
    cover = rows[0]
    assert cover["s_ref_during_min"] < 0.6 < cover["s_ref_before_mean"]


def lfw_face(name):
    path = LFW / name / f"{name}_0001.jpg"
    if not path.exists():
        pytest.skip("немає LFW (python scripts/eval_lfw.py --download)")
    return cv2.imread(str(path))


def write_video(path, face, seconds=6.0, fps=30):
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (640, 480))
    big = cv2.resize(face, (400, 400))
    for i in range(int(seconds * fps)):
        f = np.full((480, 640, 3), 90, np.uint8)
        dx = int(30 * np.sin(i / 15))
        f[40:440, 120 + dx:520 + dx] = big
        w.write(f)
    w.release()


@pytest.mark.skipif(not MTCNN.available(MODELS) or not (MODELS / "mobilefacenet_int8.onnx").exists(),
                    reason="немає моделей (python scripts/export_models.py)")
def test_import_recording_writes_encrypted_sessions(tmp_path):
    pytest.importorskip("mediapipe")
    from scripts.import_remote import import_recording

    video = tmp_path / "P09.mp4"
    write_video(video, lfw_face("George_W_Bush"))
    key = base64.b64decode(generate_key())
    ann = copy.deepcopy(ANN)
    rec_dir = import_recording(video, ann, out=tmp_path / "rec", enrolled_dir=tmp_path / "enr", key=key,
                               log=lambda *_: None)
    assert (tmp_path / "enr" / "P09.npy.enc").exists()
    assert (rec_dir / "video.mp4.enc").exists() and (rec_dir / "annotation.json.enc").exists()
    sessions = sorted(rec_dir.glob("*_s01/session.json.enc"))
    assert len(sessions) == 1
    rec = read_json_encrypted(sessions[0], key)
    assert rec["metadata"]["source"] == "remote_browser"
    assert rec["decision"]["explanation"]["reference_found"]
    s_ref = [v for v in rec["decision"]["explanation"]["series"]["s_ref"] if v is not None]
    assert min(s_ref) > 0.8      # статичне фото – ідентичність не змінюється

    # атака без еталона цільової особи не імпортується
    with pytest.raises(ValueError, match="еталона"):
        import_recording(video, ann, presentation_type="attack", tool="deep_live_cam", target="P10",
                         out=tmp_path / "rec", enrolled_dir=tmp_path / "enr", key=key, log=lambda *_: None)

    # синтетичне цільове обличчя: еталон береться з фото і зберігається зашифрованим
    photo = tmp_path / "T01.jpg"
    cv2.imwrite(str(photo), cv2.resize(lfw_face("Colin_Powell"), (400, 400)))
    att_dir = import_recording(video, ann, presentation_type="attack", tool="deep_live_cam", target="T01",
                               target_image=photo, out=tmp_path / "rec2", enrolled_dir=tmp_path / "enr",
                               key=key, log=lambda *_: None)
    assert (tmp_path / "enr" / "T01.npy.enc").exists()
    att = read_json_encrypted(next(att_dir.glob("*_s01/session.json.enc")), key)
    assert att["metadata"]["source"] == "offline_swap" and att["metadata"]["claimed_identity"] == "T01"
    s_enr = [v for v in att["decision"]["explanation"]["series"]["s_enr"] if v is not None]
    assert max(s_enr) < 0.5      # обличчя в кадрі не збігається з заявленою особою


@pytest.mark.skipif(not MTCNN.available(MODELS), reason="немає моделей MTCNN")
def test_import_rejects_video_without_face(tmp_path):
    pytest.importorskip("mediapipe")
    from scripts.import_remote import import_recording

    video = tmp_path / "empty.mp4"
    w = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 30, (320, 240))
    for _ in range(180):
        w.write(np.full((240, 320, 3), 128, np.uint8))
    w.release()
    with pytest.raises(ValueError, match="контроль|кадрів"):
        import_recording(video, copy.deepcopy(ANN), out=tmp_path / "rec", enrolled_dir=tmp_path / "enr",
                         key=base64.b64decode(generate_key()), log=lambda *_: None)
    assert not (tmp_path / "rec").exists()


def test_norm_crop_matrix_maps_landmarks_to_template():
    from pad.embedder import ARCFACE_TEMPLATE
    from pad.swapper import norm_crop_matrix

    m0 = np.array([[1.7, 0.2, 40.0], [-0.2, 1.7, 25.0]])
    pts = (ARCFACE_TEMPLATE - m0[:, 2]) @ np.linalg.inv(m0[:, :2]).T   # точки в кадрі
    m = norm_crop_matrix(pts, 112)
    mapped = pts @ m[:, :2].T + m[:, 2]
    assert np.allclose(mapped, ARCFACE_TEMPLATE, atol=1e-3)
    m128 = norm_crop_matrix(pts, 128)
    mapped = pts @ m128[:, :2].T + m128[:, 2]
    assert np.allclose(mapped[:, 0], ARCFACE_TEMPLATE[:, 0] * 128 / 128 + 8, atol=1e-3)


def test_paste_back_changes_only_face_region():
    from pad.swapper import paste_back

    frame = np.full((200, 200, 3), 50, np.uint8)
    m = np.array([[1.0, 0, -36], [0, 1.0, -36]])        # фрагмент 128×128 з (36, 36)
    fake = np.full((128, 128, 3), 250, np.uint8)
    out = paste_back(frame, frame[36:164, 36:164], fake, m)
    assert out.shape == frame.shape
    assert out[100, 100, 0] > 200        # центр обличчя – згенерований
    assert out[5, 5, 0] == 50            # фон без змін


@pytest.mark.skipif(not (SWAP / "inswapper_128.onnx").exists() or not MTCNN.available(MODELS),
                    reason="немає inswapper_128 (завантажується при першому запуску generate_attacks.py)")
def test_swap_changes_identity_to_target():
    from pad.swapper import ArcFace, InSwapper
    from scripts.generate_attacks import largest_face

    det = MTCNN(MODELS)
    a = cv2.resize(lfw_face("George_W_Bush"), (400, 400))
    b = cv2.resize(lfw_face("Colin_Powell"), (400, 400))
    arc = ArcFace(SWAP / "w600k_r50.onnx", ["CPUExecutionProvider"])
    sw = InSwapper(SWAP / "inswapper_128.onnx", ["CPUExecutionProvider"])
    fa, fb = largest_face(det, a), largest_face(det, b)
    ea, eb = arc.embed(a, fa.landmarks5), arc.embed(b, fb.landmarks5)
    swapped = sw.swap(a, fa.landmarks5, sw.latent(eb))
    es = arc.embed(swapped, largest_face(det, swapped).landmarks5)
    assert es @ eb > 0.6 and es @ ea < 0.3

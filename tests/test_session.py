"""Тести стан-машини сеансу (рисунок 3.11) на синтетичних кадрах."""

import numpy as np
import pytest

from pad.classifiers import AttackClassifier
from pad.features import l2_normalize
from pad.session import SessionManager
from pad.settings import PadSettings
from pad.structures import FrameData, SessionState
from pad.synthetic import embedding_with_similarity, hand_over_face, synthetic_face, synthetic_hand

FPS = 30.0
S = SessionState


class ConstClassifier(AttackClassifier):
    kind = "const"

    def __init__(self, value):
        self.value = value
        self.version = f"const-{value}"

    def predict_batch(self, X):
        assert X.shape[1] == 26
        return np.full(len(X), self.value)


class BrokenClassifier(AttackClassifier):
    def predict_batch(self, X):
        raise RuntimeError("модель недоступна")


class Actor:
    """Синтетичний користувач, який виконує (або ні) поточний виклик."""

    def __init__(self, perform=True, face=True, seed=0):
        self.rng = np.random.default_rng(seed)
        self.identity = l2_normalize(self.rng.normal(size=128))
        self.enrolled = embedding_with_similarity(self.identity, 0.8, self.rng)
        self.perform = perform
        self.face = face

    def frame(self, t, session):
        if not self.face:
            return FrameData(t=t, size=(1280, 720))
        face, hand, s = synthetic_face(), None, 0.97
        ch = session.current
        if session.state == S.CHALLENGE and ch is not None and self.perform:
            dt = t - (ch.t_start if ch.t_start is not None else t)
            if dt > 0.4:
                if ch.type.startswith("cover_"):
                    hand, s = hand_over_face(face, ch.side, 1.3), 0.6
                elif ch.type == "turn_head":
                    face, s = synthetic_face(yaw=40 if ch.direction == "left" else -40), 0.75
                else:
                    fw = float(np.linalg.norm(face[454] - face[234]))
                    d = -0.6 + 1.2 * min(1.0, (dt - 0.4) / 0.8)
                    hand = synthetic_hand((face[1, 0] + d * fw, face[1, 1]), 0.5 * fw, 0.8 * fw)
                    s = 0.85
        emb = embedding_with_similarity(self.identity, s, self.rng)
        return FrameData(t=t, embedding=emb, det_score=0.99, face=face, hand=hand, size=(1280, 720))


def run(manager, actor, max_s=70.0):
    session = manager.start_session(person_id=1, operator_id=1, enrolled_embedding=actor.enrolled)
    for i in range(int(max_s * FPS)):
        manager.on_frame(session.id, actor.frame(i / FPS, session))
        if session.state.is_final:
            break
    return session


def states(session):
    return [st for _, st in session.transitions]


def test_bona_fide_flow_follows_state_diagram():
    s = run(SessionManager(ConstClassifier(0.1)), Actor())
    assert s.state == S.BONA_FIDE
    seq = states(s)
    assert seq[:4] == [S.CREATED, S.REFERENCE, S.GENERATING, S.CHALLENGE]
    assert seq[-2:] == [S.CLASSIFYING, S.BONA_FIDE]
    assert seq.count(S.CHALLENGE) == seq.count(S.VERIFYING) == seq.count(S.FEATURES) == 3
    assert s.attempt_no == 1
    d = s.decision
    assert d.decision == "bona_fide" and d.score == pytest.approx(0.1) and d.is_attack is False
    assert len(d.explanation["attempts"][-1]) == 3
    assert d.explanation["duration_s"] <= 20.0
    assert s.decision_latency_ms < 1000.0


def test_features_show_drift_during_challenge():
    s = run(SessionManager(ConstClassifier(0.1)), Actor())
    for ch in s.challenges:
        f = ch.features.as_dict()
        assert len(f) == 26
        assert f["s_ref_during_min"] < f["s_ref_before_mean"] - 0.05
        assert f["s_ref_after_mean"] > f["s_ref_during_min"]


def test_attack_decision():
    s = run(SessionManager(ConstClassifier(0.9)), Actor())
    assert s.state == S.ATTACK
    assert s.decision.is_attack and s.decision.score == pytest.approx(0.9)


def test_threshold_from_settings():
    s = run(SessionManager(ConstClassifier(0.6), PadSettings(threshold=0.7)), Actor())
    assert s.state == S.BONA_FIDE


def test_not_performed_retries_twice_then_fails():
    s = run(SessionManager(ConstClassifier(0.1)), Actor(perform=False))
    assert s.state == S.FAILED and s.failure_reason == "not_performed"
    assert s.attempt_no == 3
    assert states(s).count(S.GENERATING) == 3
    assert s.decision.decision == "failed" and s.decision.is_attack is None
    assert len(s.decision.explanation["attempts"]) == 3


def test_no_face_fails_without_positive_decision():
    s = run(SessionManager(ConstClassifier(0.0)), Actor(face=False))
    assert s.state == S.FAILED and s.failure_reason == "no_face"


def test_module_error_fails_session():
    s = run(SessionManager(BrokenClassifier()), Actor())
    assert s.state == S.FAILED and s.failure_reason == "error"
    assert "модель недоступна" in s.decision.explanation["detail"]


def test_session_timeout():
    s = run(SessionManager(ConstClassifier(0.1), PadSettings(session_timeout_s=4.0)), Actor())
    assert s.state == S.FAILED and s.failure_reason == "timeout"


def test_frames_after_final_state_are_ignored():
    manager = SessionManager(ConstClassifier(0.1))
    actor = Actor()
    s = run(manager, actor)
    n = len(s.t)
    manager.on_frame(s.id, actor.frame(100.0, s))
    assert len(s.t) == n and s.state == S.BONA_FIDE


def test_decision_json_is_valid():
    import json

    s = run(SessionManager(ConstClassifier(0.1)), Actor())
    d = json.loads(s.decision.to_json())
    assert d["decision"] == "bona_fide"
    assert len(d["explanation"]["series"]["s_ref"]) == len(s.t)


def test_dataset_rows_from_recorded_session(tmp_path):
    import json

    from pad.dataset import read_csv, rows_from_recording, write_csv

    s = run(SessionManager(ConstClassifier(0.1)), Actor())
    rec = {"metadata": {"participant": "P01", "presentation_type": "attack", "swap_tool": "deep_live_cam"},
           "settings": PadSettings().to_dict(), "decision": json.loads(s.decision.to_json())}
    path = tmp_path / "P01_attack" / "session.json"
    path.parent.mkdir()
    path.write_text(json.dumps(rec), encoding="utf-8")
    rows = rows_from_recording(path)
    assert len(rows) == 3
    assert all(r["label"] == 1 and r["pai_species"] == "deep_live_cam" for r in rows)
    for r, ch in zip(rows, s.challenges):
        assert r["challenge_type"] == ch.type
        assert r["s_ref_during_min"] == pytest.approx(float(ch.features.as_dict()["s_ref_during_min"]), abs=1e-3)
    write_csv(rows, tmp_path / "f.csv")
    assert len(read_csv(tmp_path / "f.csv")) == 3

"""Контроль якості записів пілотного набору (decrypt_recording.py --qc, протокол збору даних, розділ 7)."""

import importlib.util
import json
import os
from pathlib import Path

import pytest

from pad.settings import CHALLENGE_TYPES
from pad.storage import write_encrypted
from test_storage import script

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("decrypt_recording", ROOT / "scripts" / "decrypt_recording.py")
dr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dr)


@pytest.fixture
def key():
    return os.urandom(32)


def series(n=150, missing=(), fps=30.0, gap_at=None, gap_s=0.0):
    """Ряди t і s_ref: n кадрів, у кадрах missing обличчя немає, після кадру gap_at – перерва gap_s."""
    t = [round(i / fps + (gap_s if gap_at is not None and i > gap_at else 0.0), 4) for i in range(n)]
    s_ref = [None if i in missing else 0.9 for i in range(n)]
    return {"t": t, "s_ref": s_ref, "s_enr": list(s_ref)}


def make_rec(participant="P01", kind="bona_fide", attempts=(CHALLENGE_TYPES[:3],), ser=None,
             tool="", target="", lighting="normal"):
    """Синтетичний session.json: метадані, спроби з типами викликів і ряди series."""
    return {"metadata": {"participant": participant, "presentation_type": kind, "swap_tool": tool,
                         "target_face": target, "conditions": {"lighting": lighting}},
            "decision": {"decision": kind, "score": 0.5, "threshold": 0.5,
                         "explanation": {"attempts": [[{"challenge_type": c, "performed": True} for c in a]
                                                      for a in attempts],
                                         "series": ser or series()}}}


def save(tmp_path, key, rec, stamp="120000"):
    """Тека сеансу із зашифрованим session.json.enc, як після record_session.py."""
    meta = rec["metadata"]
    folder = tmp_path / "recordings" / f"{meta['participant']}_{meta['presentation_type']}_20261017_{stamp}"
    folder.mkdir(parents=True)
    return write_encrypted(folder / "session.json", json.dumps(rec).encode("utf-8"), key)


def test_face_share_after_reference():
    assert dr.face_share([None, None, 0.9, None, 0.8, 0.85]) == pytest.approx(0.75)
    assert dr.face_share([0.9] * 10) == 1.0
    assert dr.face_share([None] * 5) == 0.0 and dr.face_share([]) == 0.0


def test_max_gap():
    assert dr.max_gap([0.0, 0.0333, 0.0667, 0.3]) == pytest.approx(0.2333)
    assert dr.max_gap([0.0, 0.1, 0.3]) == 0.2                  # рівно 0,2 с – ще не перерва
    assert dr.max_gap([0.0]) == 0.0 and dr.max_gap([]) == 0.0


def test_quality_pass_and_fail():
    q = dr.quality(make_rec(ser=series(100, missing={0, 1, 2, 3, 50})))  # 95 з 96 кадрів після еталона
    assert q["passed"] and q["frames"] == 100 and q["face_share"] == pytest.approx(95 / 96)
    assert q["swap_tool"] == "-" and q["lighting"] == "normal"
    q = dr.quality(make_rec(ser=series(100, missing=set(range(40, 46)))))
    assert not q["passed"] and q["face_share"] == pytest.approx(0.94)
    assert "кадрів з обличчям 94,0 % < 95 %" in q["problems"]
    q = dr.quality(make_rec("P02", "attack", tool="deep_live_cam", target="T01", lighting="low",
                            ser=series(gap_at=60, gap_s=0.25)))
    assert not q["passed"] and q["max_gap"] == pytest.approx(0.2833, abs=1e-4)
    assert q["problems"] == ["перерва 0,283 с > 0,2 с"]
    assert (q["swap_tool"], q["target_face"], q["lighting"]) == ("deep_live_cam", "T01", "low")
    lines = dr.quality_lines(q)
    assert "інструмент deep_live_cam, ціль T01, освітлення low" in lines[0] and lines[-1].startswith("  FAIL –")


def test_challenge_counts_per_participant_and_type():
    recs = [make_rec("P01", "bona_fide", attempts=(CHALLENGE_TYPES[:3], CHALLENGE_TYPES[2:])),
            make_rec("P01", "bona_fide", attempts=(CHALLENGE_TYPES[:3],))]
    counts = dr.challenge_counts(recs)
    assert list(counts) == [("P01", "attack"), ("P01", "bona_fide")]   # атак немає – нулі
    assert counts[("P01", "bona_fide")] == {"cover_left": 2, "cover_right": 2, "cover_mouth": 3,
                                            "turn_head": 1, "hand_swipe": 1}
    assert all(v == 0 for v in counts[("P01", "attack")].values())
    lines, ok = dr.counts_lines(counts)
    assert not ok
    assert "  P01 bona_fide: cover_left 2 (мало), cover_right 2 (мало), cover_mouth 3, " \
           "turn_head 1 (мало), hand_swipe 1 (мало) – FAIL" in lines
    full = [make_rec("P01", kind, attempts=[CHALLENGE_TYPES] * 3) for kind in ("bona_fide", "attack")]
    lines, ok = dr.counts_lines(dr.challenge_counts(full))
    assert ok and not any("FAIL" in line for line in lines)


def test_qc_script_on_encrypted_recordings(tmp_path, key):
    save(tmp_path, key, make_rec("P01", "bona_fide", attempts=[CHALLENGE_TYPES] * 3))
    save(tmp_path, key, make_rec("P01", "attack", attempts=[CHALLENGE_TYPES] * 2, tool="deep_live_cam",
                                 target="T01", ser=series(missing=set(range(50, 60)))))
    before = sorted(p.relative_to(tmp_path) for p in tmp_path.rglob("*"))
    res = script("decrypt_recording.py", tmp_path / "recordings", "--qc", key=key)
    assert res.returncode == 1, res.stderr
    out = res.stdout
    assert "  PASS" in out and "FAIL – кадрів з обличчям 93,3 % < 95 %" in out
    assert "інструмент deep_live_cam, ціль T01" in out
    assert "  P01 attack: cover_left 2 (мало)" in out and "hand_swipe 2 (мало) – FAIL" in out
    assert "  P01 bona_fide: cover_left 3, cover_right 3, cover_mouth 3, turn_head 3, hand_swipe 3\n" in out
    assert "сеансів 2, з них FAIL 1" in out and chr(0x2014) not in out
    assert sorted(p.relative_to(tmp_path) for p in tmp_path.rglob("*")) == before   # на диск нічого не пишеться


def test_qc_script_all_pass(tmp_path, key):
    save(tmp_path, key, make_rec("P01", "bona_fide", attempts=[CHALLENGE_TYPES] * 3))
    save(tmp_path, key, make_rec("P01", "attack", attempts=[CHALLENGE_TYPES] * 3, tool="deepfacelive"))
    res = script("decrypt_recording.py", tmp_path / "recordings", "--qc", key=key)
    assert res.returncode == 0, res.stdout + res.stderr
    assert "FAIL" not in res.stdout.replace("FAIL 0", "") and "сеансів 2, з них FAIL 0" in res.stdout
    res = script("decrypt_recording.py", tmp_path / "recordings", "--qc", "--full", key=key)
    assert res.returncode != 0 and "--qc" in res.stderr
    res = script("decrypt_recording.py", tmp_path / "recordings", "--qc", key=os.urandom(32))
    assert res.returncode == 1 and "неправильний ключ" in res.stderr

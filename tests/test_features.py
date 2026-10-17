"""Тести ознак дрейфу ембединга на синтетичних рядах (FR-53, FR-57)."""

import numpy as np
import pytest

from pad.features import (FEATURE_NAMES, DriftFeatureExtractor, InsufficientDataError,
                          l2_normalize, reference_embedding)
from pad.synthetic import embedding_with_similarity

FPS = 30.0


def series(before, during, after, during_s=2.0, before_s=1.0, after_s=1.5):
    """Ряд з t_start = 0 і t_end = during_s; аргументи – функції часу або константи."""
    t = np.round(np.arange(-before_s, during_s + after_s + 1e-9, 1 / FPS), 6)
    f = [before, during, after]
    s = np.empty_like(t)
    for i, tt in enumerate(t):
        part = 0 if tt < 0 else (1 if tt <= during_s else 2)
        s[i] = f[part](tt) if callable(f[part]) else f[part]
    return t, s


def extract(t, s_ref, s_enr=None, t_end=2.0):
    ext = DriftFeatureExtractor(before_s=1.0, after_s=1.5, recovery_ratio=0.9)
    return ext.extract_series(t, s_ref, s_ref if s_enr is None else s_enr, 0.0, t_end).as_dict()


def test_feature_names():
    assert len(FEATURE_NAMES) == 26 == 2 * (3 * 4 + 1)
    assert len(set(FEATURE_NAMES)) == 26
    assert FEATURE_NAMES[0] == "s_ref_before_min" and FEATURE_NAMES[12] == "s_ref_trec"
    assert FEATURE_NAMES[13] == "s_enr_before_min" and FEATURE_NAMES[-1] == "s_enr_trec"


def test_window_stats_known_values():
    st = DriftFeatureExtractor.window_stats(np.array([1.0, 0.8, 0.9, np.nan]))
    assert st["min"] == pytest.approx(0.8)
    assert st["mean"] == pytest.approx(0.9)
    assert st["var"] == pytest.approx(np.var([1.0, 0.8, 0.9]))
    assert st["jump"] == pytest.approx(0.2)


def test_constant_series():
    t, s = series(0.95, 0.95, 0.95)
    f = extract(t, s)
    for w in ("before", "during", "after"):
        assert f[f"s_ref_{w}_min"] == pytest.approx(0.95)
        assert f[f"s_ref_{w}_var"] == pytest.approx(0.0, abs=1e-12)
        assert f[f"s_ref_{w}_jump"] == pytest.approx(0.0, abs=1e-12)
    assert f["s_ref_trec"] == 0.0


def test_abrupt_jump_vs_smooth_dip():
    t, smooth = series(0.95, lambda x: 0.95 - 0.4 * np.sin(np.pi * x / 2.0), 0.95)
    t, abrupt = series(0.95, lambda x: 0.55 if 0.8 < x < 1.2 else 0.95, 0.95)
    fs, fa = extract(t, smooth), extract(t, abrupt)
    assert fs["s_ref_during_min"] == pytest.approx(0.55, abs=0.01)
    assert fa["s_ref_during_jump"] == pytest.approx(0.4, abs=1e-6)
    assert fs["s_ref_during_jump"] < 0.05
    assert fa["s_ref_during_jump"] > 5 * fs["s_ref_during_jump"]


def test_recovery_time_formula_3_4():
    # після виклику подібність лінійно зростає від 0,5 до 0,95 за 1 с
    t, s = series(0.95, 0.5, lambda x: min(0.95, 0.5 + 0.45 * (x - 2.0)))
    f = extract(t, s)
    expected = (0.9 * 0.95 - 0.5) / 0.45  # момент, коли s(t) >= 0,9 · s_до
    assert f["s_ref_trec"] == pytest.approx(expected, abs=1 / FPS + 1e-9)


def test_no_recovery_gives_window_length():
    t, s = series(0.95, 0.5, 0.6)
    assert extract(t, s)["s_ref_trec"] == pytest.approx(1.5)


def test_enr_series_is_independent():
    t, s_ref = series(0.95, 0.6, 0.95)
    _, s_enr = series(0.75, 0.2, 0.75)
    f = extract(t, s_ref, s_enr)
    assert f["s_enr_before_mean"] == pytest.approx(0.75)
    assert f["s_enr_during_min"] == pytest.approx(0.2)
    assert f["s_ref_during_min"] == pytest.approx(0.6)


def test_missing_embeddings_are_skipped():
    t, s = series(0.95, 0.6, 0.95)
    s[::3] = np.nan
    f = extract(t, s)
    assert np.isfinite(list(f.values())).all()


def test_insufficient_data_raises():
    t, s = series(0.95, 0.6, 0.95)
    s[t < 0] = np.nan
    with pytest.raises(InsufficientDataError):
        extract(t, s)


def test_reference_embedding_formula_3_1(rng):
    base = l2_normalize(rng.normal(size=128))
    embs = [embedding_with_similarity(base, 0.95, rng) for _ in range(10)]
    scores = [0.99] * 8 + [0.5, 0.3]
    ref = reference_embedding(embs, scores, 0.9)
    expected = l2_normalize(np.sum(embs[:8], axis=0))
    assert np.allclose(ref, expected, atol=1e-6)
    assert np.linalg.norm(ref) == pytest.approx(1.0, abs=1e-6)
    assert reference_embedding(embs, [0.1] * 10, 0.9) is None


def test_similarity_is_cosine(rng):
    a, b = rng.normal(size=128), rng.normal(size=128)
    cos = float(np.dot(a, b) / np.linalg.norm(a) / np.linalg.norm(b))
    assert DriftFeatureExtractor.similarity(a * 3, b * 0.5) == pytest.approx(cos, abs=1e-6)
    e = embedding_with_similarity(a, 0.7, rng)
    assert DriftFeatureExtractor.similarity(e, a) == pytest.approx(0.7, abs=1e-5)

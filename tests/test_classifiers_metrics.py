"""Тести класифікаторів і показників ДСТУ ISO/IEC 30107-3:2023."""

import numpy as np
import pytest

from pad.classifiers import (LogisticClassifier, ThresholdClassifier, load_classifier,
                             session_score)
from pad.dataset import to_arrays
from pad.features import FEATURE_NAMES
from pad.metrics import (apcer_bpcer, auc, bpcer_at_apcer, eer, evaluate, rates_from_decisions,
                         threshold_for_bpcer)
from pad.synthetic import make_synthetic_dataset


@pytest.fixture(scope="module")
def data():
    return to_arrays(make_synthetic_dataset(n_subjects=4, sessions_per_class=6, seed=7))


# ----- класифікатори -----

def test_threshold_classifier_is_monotone_and_bounded():
    clf = ThresholdClassifier("s_ref_during_min", threshold=0.6, scale=0.05)
    X = np.zeros((5, 26))
    X[:, FEATURE_NAMES.index("s_ref_during_min")] = [0.2, 0.5, 0.6, 0.7, 0.95]
    p = clf.predict_batch(X)
    assert np.all((p >= 0) & (p <= 1))
    assert np.all(np.diff(p) < 0)
    assert p[2] == pytest.approx(0.5)


def test_threshold_fit_separates(data):
    clf = ThresholdClassifier().fit(data["X"], data["y"])
    assert auc(clf.predict_batch(data["X"]), data["y"]) > 0.7


def test_logistic_learns_synthetic_data(data):
    clf = LogisticClassifier().fit(data["X"], data["y"])
    p = clf.predict_batch(data["X"])
    assert np.all((p >= 0) & (p <= 1))
    assert auc(p, data["y"]) > 0.9
    assert len(clf.weights) == 26
    assert clf.importance()[0][0] in FEATURE_NAMES


def test_folded_weights_match_sklearn_pipeline(data):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    clf = LogisticClassifier(C=0.5).fit(data["X"], data["y"])
    pipe = make_pipeline(StandardScaler(), LogisticRegression(C=0.5, class_weight="balanced",
                                                              max_iter=5000, random_state=0))
    pipe.fit(data["X"], data["y"])
    assert np.allclose(clf.predict_batch(data["X"]), pipe.predict_proba(data["X"])[:, 1], atol=1e-6)


def test_json_round_trip(tmp_path, data):
    for clf in (LogisticClassifier().fit(data["X"], data["y"]),
                ThresholdClassifier().fit(data["X"], data["y"])):
        path = tmp_path / f"{clf.kind}.json"
        clf.save(path)
        loaded = load_classifier(path)
        assert type(loaded) is type(clf)
        assert np.allclose(loaded.predict_batch(data["X"]), clf.predict_batch(data["X"]))


def test_onnx_round_trip(tmp_path, data):
    pytest.importorskip("onnx")
    clf = LogisticClassifier().fit(data["X"], data["y"])
    clf.export_onnx(tmp_path / "m.onnx")
    loaded = load_classifier(tmp_path / "m.onnx")
    assert loaded.version == clf.version
    assert np.allclose(loaded.predict_batch(data["X"]), clf.predict_batch(data["X"]), atol=1e-5)


def test_session_score_is_maximum():
    assert session_score([0.1, 0.8, 0.3]) == 0.8


# ----- показники -----

BONA = [0.1, 0.2, 0.3, 0.6]
ATTACK = [0.4, 0.7, 0.8, 0.9]
SCORES = BONA + ATTACK
LABELS = [0] * 4 + [1] * 4


def test_apcer_bpcer_at_threshold():
    r = apcer_bpcer(SCORES, LABELS, 0.5)
    assert r["bpcer"] == pytest.approx(0.25)   # 0,6 >= 0,5
    assert r["apcer"] == pytest.approx(0.25)   # 0,4 < 0,5
    assert r["acer"] == pytest.approx(0.25)


def test_apcer_per_species_and_maximum():
    species = [""] * 4 + ["dlc", "dlc", "dfl", "dfl"]
    r = apcer_bpcer(SCORES, LABELS, 0.75, species)
    assert r["apcer_per_species"] == {"dfl": 0.0, "dlc": 1.0}
    assert r["apcer"] == 1.0


def test_perfect_separation():
    s, y = [0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1]
    assert eer(s, y)[0] == pytest.approx(0.0)
    assert auc(s, y) == 1.0
    assert bpcer_at_apcer(s, y, 0.05)[0] == 0.0


def test_eer_overlapping():
    e, thr = eer(SCORES, LABELS)
    assert e == pytest.approx(0.25, abs=1e-9)
    r = apcer_bpcer(SCORES, LABELS, thr)
    assert abs(r["apcer"] - r["bpcer"]) <= 0.25


def test_bpcer20():
    bpcer, thr = bpcer_at_apcer(SCORES, LABELS, 0.05)
    assert thr == pytest.approx(0.4)        # жодна атака не пропущена
    assert bpcer == pytest.approx(0.25)     # 0,6 >= 0,4


def test_threshold_for_bpcer():
    thr = threshold_for_bpcer(SCORES, LABELS, target=0.0)
    assert np.mean(np.array(BONA) >= thr) == 0.0
    thr = threshold_for_bpcer(SCORES, LABELS, target=0.25)
    assert thr == pytest.approx(0.6)


def test_rates_from_decisions_and_evaluate():
    r = rates_from_decisions([0, 0, 1, 1, 0, 1, 1, 1], LABELS)
    assert r["bpcer"] == 0.5 and r["apcer"] == 0.25
    full = evaluate(SCORES, LABELS, 0.5)
    assert {"apcer", "bpcer", "bpcer20", "eer", "auc"} <= set(full)

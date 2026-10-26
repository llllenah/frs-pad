"""Набір ознак викликів: читання і запис CSV, побудова з записаних сеансів.

Один рядок CSV – один виконаний виклик: учасник (псевдонім), ідентифікатор
сеансу, мітка (1 – атака), вид атаки, тип виклику та 26 ознак. Ознаки
обчислюються заново з рядів s_ref(t), s_enr(t), збережених у session.json,
тому після зміни коду ознак достатньо ще раз запустити build_dataset.py.
Записи читаються і зашифрованими (session.json.enc, ключ з PAD_DATASET_KEY),
і відкритими (session.json).
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from .features import FEATURE_NAMES, DriftFeatureExtractor, InsufficientDataError
from .storage import SUFFIX, read_json

META_COLUMNS = ["subject", "session_uid", "label", "pai_species", "target", "challenge_type", "order", "attempt"]


def write_csv(rows: list[dict], path: str | Path) -> None:
    """Записує рядки набору у CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=META_COLUMNS + FEATURE_NAMES, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({"attempt": 1, **r})


def read_csv(path: str | Path) -> list[dict]:
    """Читає набір з CSV."""
    with open(path, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        r["label"] = int(r["label"])
        for name in FEATURE_NAMES:
            r[name] = float(r[name])
    return rows


def to_arrays(rows: list[dict]) -> dict[str, np.ndarray]:
    """Матриця ознак X, мітки y та групувальні поля."""
    return {
        "X": np.array([[r[n] for n in FEATURE_NAMES] for r in rows], dtype=float),
        "y": np.array([r["label"] for r in rows], dtype=int),
        "subject": np.array([r["subject"] for r in rows]),
        "session": np.array([r["session_uid"] for r in rows]),
        "species": np.array([r.get("pai_species") or "" for r in rows]),
        "target": np.array([r.get("target") or "" for r in rows]),
        "challenge_type": np.array([r["challenge_type"] for r in rows]),
    }


def rows_from_recording(path: str | Path, extractor: DriftFeatureExtractor | None = None,
                        key: bytes | None = None) -> list[dict]:
    """Рядки набору з одного файлу session.json або session.json.enc, записаного record_session.py."""
    rec = read_json(path, key)
    meta, dec = rec["metadata"], rec["decision"]
    expl = dec["explanation"]
    st = rec.get("settings", {})
    extractor = extractor or DriftFeatureExtractor(st.get("before_s", 1.0), st.get("after_s", 1.5),
                                                   st.get("recovery_ratio", 0.9))
    t0 = expl.get("t0") or 0.0
    ser = expl["series"]
    t = np.array(ser["t"], dtype=float)
    nan = float("nan")
    s_ref = np.array([nan if v is None else v for v in ser["s_ref"]], dtype=float)
    s_enr = np.array([nan if v is None else v for v in ser["s_enr"]], dtype=float)
    label = 1 if meta["presentation_type"] == "attack" else 0
    uid = Path(path).parent.name
    rows = []
    for attempt, challenges in enumerate(expl.get("attempts", []), start=1):
        for order, ch in enumerate(challenges):
            if not ch.get("performed") or ch.get("t_start") is None or ch.get("t_end") is None:
                continue
            try:
                fv = extractor.extract_series(t, s_ref, s_enr, ch["t_start"] - t0, ch["t_end"] - t0)
            except InsufficientDataError:
                continue
            rows.append({"subject": meta["participant"], "session_uid": uid, "label": label,
                         "pai_species": meta.get("swap_tool", "") if label else "",
                         "target": meta.get("target_face", "") if label else "",
                         "challenge_type": ch["challenge_type"], "order": order, "attempt": attempt,
                         **dict(zip(FEATURE_NAMES, map(float, fv.values)))})
    return rows


def recording_files(root: str | Path) -> list[Path]:
    """Файли сеансів у теці root (рекурсивно), по одному на теку; .enc має перевагу."""
    found: dict[Path, Path] = {}
    for name in ("session.json", "session.json" + SUFFIX):
        for path in Path(root).rglob(name):
            found[path.parent] = path
    return [found[d] for d in sorted(found)]


def build_from_recordings(root: str | Path, key: bytes | None = None) -> list[dict]:
    """Рядки з усіх session.json і session.json.enc у теці root (рекурсивно)."""
    rows = []
    for path in recording_files(root):
        rows.extend(rows_from_recording(path, key=key))
    return rows

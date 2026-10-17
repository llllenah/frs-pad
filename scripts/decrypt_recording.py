"""Перегляд зашифрованих записів пілотного набору (video.mp4.enc, session.json.enc).

Ключ береться зі змінної середовища PAD_DATASET_KEY (32 байти в base64).
Без -o нічого не записується на диск: виводиться лише короткий зміст сеансу.
Приклади:
    # новий ключ для PAD_DATASET_KEY (зберігати окремо від data/, наприклад у менеджері паролів)
    python scripts/decrypt_recording.py --new-key
    # метадані, виклики, оцінка і рішення всіх сеансів у теці
    python scripts/decrypt_recording.py data/recordings
    # повний session.json одного сеансу в консоль
    python scripts/decrypt_recording.py data/recordings/P01_bona_fide_20261017_120000 --full
    # контроль якості за протоколом збору даних (розділ 7): обличчя, перерви, кількість викликів
    python scripts/decrypt_recording.py data/recordings --qc
    # розшифрувати video.mp4 і session.json для перегляду (після перегляду теку видалити)
    python scripts/decrypt_recording.py data/recordings/P01_bona_fide_20261017_120000 -o view
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pad.cli import setup_console  # noqa: E402
from pad.settings import CHALLENGE_TYPES  # noqa: E402
from pad.storage import (SUFFIX, DatasetKeyError, DecryptionError, decrypt_file,  # noqa: E402
                         generate_key, load_key, plain_name, read_json_encrypted)

#: Вимоги протоколу збору даних (розділ 7).
MIN_FACE_SHARE = 0.95      # частка кадрів з обличчям після фіксації еталона
MAX_GAP_S = 0.2            # найбільша допустима перерва в series.t, с
MIN_CHALLENGES = 3         # кожного типу виклику на учасника в справжніх поданнях і в атаках
PRESENTATION_TYPES = ("bona_fide", "attack")


def recordings(paths: list[str]) -> dict[Path, list[Path]]:
    """Зашифровані файли, згруповані за текою сеансу."""
    found: dict[Path, list[Path]] = {}
    for p in map(Path, paths):
        files = [p] if p.is_file() else sorted(p.rglob("*" + SUFFIX))
        for f in files:
            found.setdefault(f.parent, []).append(f)
    return found


def summary(rec: dict) -> list[str]:
    """Короткий зміст session.json: хто, що подавалося, виклики і рішення."""
    meta, dec = rec["metadata"], rec["decision"]
    expl = dec["explanation"]
    kind = meta["presentation_type"]
    if kind == "attack":
        kind += f" ({meta.get('swap_tool')}, ціль {meta.get('target_face') or '-'})"
    lines = [f"  учасник {meta['participant']}, заявлена особа {meta.get('claimed_identity')}, подання {kind}",
             f"  умови: {meta.get('conditions', {})}",
             f"  рішення {dec['decision']}, оцінка {dec['score']}, поріг {dec['threshold']}, "
             f"тривалість {expl.get('duration_s')} с, спроб {len(expl.get('attempts', []))}"]
    for no, attempt in enumerate(expl.get("attempts", []), start=1):
        for ch in attempt:
            mark = "виконано" if ch.get("performed") else "не виконано"
            lines.append(f"    спроба {no}: {ch['challenge_type']} – {mark}, p = {ch.get('score')}")
    return lines


def num(x: float, digits: int) -> str:
    """Число з десятковою комою, як у тексті роботи."""
    return f"{x:.{digits}f}".replace(".", ",")


def face_share(s_ref: list) -> float:
    """Частка кадрів з обличчям: ненульові s_ref від першого ненульового (еталон зафіксовано)."""
    first = next((i for i, v in enumerate(s_ref) if v is not None), None)
    if first is None:
        return 0.0
    rest = s_ref[first:]
    return sum(v is not None for v in rest) / len(rest)


def max_gap(t: list) -> float:
    """Найбільша перерва між сусідніми мітками series.t, с."""
    ts = [v for v in t if v is not None]
    return round(max((b - a for a, b in zip(ts, ts[1:])), default=0.0), 4)


def quality(rec: dict) -> dict:
    """Перевірка одного сеансу: частка кадрів з обличчям, перерви в ряді, умови запису."""
    meta = rec["metadata"]
    series = rec["decision"]["explanation"].get("series", {})
    s_ref = series.get("s_ref", [])
    share, gap = face_share(s_ref), max_gap(series.get("t", []))
    problems = []
    if share < MIN_FACE_SHARE:
        problems.append(f"кадрів з обличчям {num(share * 100, 1)} % < {num(MIN_FACE_SHARE * 100, 0)} %")
    if gap > MAX_GAP_S:
        problems.append(f"перерва {num(gap, 3)} с > {num(MAX_GAP_S, 1)} с")
    return {"participant": meta.get("participant", "?"),
            "presentation_type": meta.get("presentation_type", "?"),
            "swap_tool": meta.get("swap_tool") or "-", "target_face": meta.get("target_face") or "-",
            "lighting": meta.get("conditions", {}).get("lighting") or "-",
            "frames": len(s_ref), "face_share": share, "max_gap": gap,
            "problems": problems, "passed": not problems}


def quality_lines(q: dict) -> list[str]:
    """Рядки звіту про якість одного сеансу."""
    return [f"  {q['participant']}, {q['presentation_type']}, інструмент {q['swap_tool']}, "
            f"ціль {q['target_face']}, освітлення {q['lighting']}",
            f"  кадрів {q['frames']}, з обличчям {num(q['face_share'] * 100, 1)} %, "
            f"найбільша перерва {num(q['max_gap'], 3)} с",
            "  PASS" if q["passed"] else "  FAIL – " + "; ".join(q["problems"])]


def challenge_counts(recs: list[dict]) -> dict[tuple[str, str], Counter]:
    """Скільки разів кожен тип виклику траплявся в спробах, за учасником і типом подання.

    Для кожного учасника є обидва типи подання: якщо одного немає, він дає нулі і позначається.
    """
    counts: dict[tuple[str, str], Counter] = {}
    for rec in recs:
        meta = rec["metadata"]
        participant = meta.get("participant", "?")
        for kind in (*PRESENTATION_TYPES, meta.get("presentation_type", "?")):
            counts.setdefault((participant, kind), Counter({t: 0 for t in CHALLENGE_TYPES}))
        c = counts[(participant, meta.get("presentation_type", "?"))]
        for attempt in rec["decision"]["explanation"].get("attempts", []):
            c.update(ch["challenge_type"] for ch in attempt)
    return dict(sorted(counts.items()))


def counts_lines(counts: dict[tuple[str, str], Counter]) -> tuple[list[str], bool]:
    """Таблиця кількості викликів і ознака, що кожного типу не менше MIN_CHALLENGES."""
    lines, ok = [f"кількість викликів (потрібно не менше {MIN_CHALLENGES} кожного типу):"], True
    for (participant, kind), c in counts.items():
        low = [t for t in CHALLENGE_TYPES if c[t] < MIN_CHALLENGES]
        ok = ok and not low
        cells = ", ".join(f"{t} {c[t]}" + (" (мало)" if t in low else "") for t in CHALLENGE_TYPES)
        lines.append(f"  {participant} {kind}: {cells}" + (" – FAIL" if low else ""))
    return lines, ok


def main() -> int:
    setup_console()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="*", help="тека сеансу, тека з сеансами або файл .enc")
    ap.add_argument("-o", "--out", default=None, help="тека для розшифрованих файлів")
    ap.add_argument("--full", action="store_true", help="вивести весь session.json")
    ap.add_argument("--qc", action="store_true", help="перевірити якість записів за протоколом збору даних")
    ap.add_argument("--new-key", action="store_true", help="згенерувати новий ключ і вийти")
    args = ap.parse_args()

    if args.new_key:
        print(generate_key())
        return 0
    if not args.paths:
        ap.error("вкажіть теку сеансу або файл .enc")
    if args.qc and (args.out is not None or args.full):
        ap.error("--qc не поєднується з -o і --full")
    try:
        key = load_key()
    except DatasetKeyError as exc:
        sys.exit(str(exc))
    found = recordings(args.paths)
    if not found:
        print("зашифрованих записів не знайдено")
        return 1
    errors, failed, recs = 0, 0, []
    for folder, files in found.items():
        print(folder.name)
        try:
            for f in files:
                if args.out is not None:
                    out = Path(args.out) / folder.name / plain_name(f)
                    out.parent.mkdir(parents=True, exist_ok=True)
                    print(f"  {f.name} -> {decrypt_file(f, out, key)}")
                elif f.name == "session.json" + SUFFIX:
                    rec = read_json_encrypted(f, key)
                    if args.qc:
                        q = quality(rec)
                        failed += not q["passed"]
                        recs.append(rec)
                        print("\n".join(quality_lines(q)))
                        continue
                    print(json.dumps(rec, ensure_ascii=False, indent=1) if args.full else "\n".join(summary(rec)))
        except DecryptionError as exc:
            print(f"  помилка: {exc}", file=sys.stderr)
            errors += 1
    if args.out is not None:
        print(f"відкриті копії в {args.out} – після перегляду видаліть теку")
    if args.qc:
        lines, counts_ok = counts_lines(challenge_counts(recs))
        print("\n".join(lines))
        print(f"сеансів {len(recs)}, з них FAIL {failed}")
        return 1 if errors or failed or not counts_ok else 0
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())

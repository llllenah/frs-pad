"""Будує CSV ознак викликів із записаних сеансів (data/recordings/*/session.json.enc).

Зашифровані записи читаються ключем зі змінної середовища PAD_DATASET_KEY.
Запуск: python scripts/build_dataset.py data/recordings -o data/features.csv
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pad.cli import setup_console  # noqa: E402
from pad.dataset import build_from_recordings, write_csv  # noqa: E402
from pad.storage import DatasetKeyError, DecryptionError  # noqa: E402


def main() -> int:
    setup_console()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root", nargs="?", default=str(ROOT / "data" / "recordings"))
    ap.add_argument("-o", "--out", default=str(ROOT / "data" / "features.csv"))
    args = ap.parse_args()
    try:
        rows = build_from_recordings(args.root)
    except (DatasetKeyError, DecryptionError) as exc:
        sys.exit(f"не вдалося прочитати записи: {exc}")
    if not rows:
        print(f"у {args.root} немає записаних сеансів з виконаними викликами")
        return 1
    write_csv(rows, args.out)
    by = Counter((r["subject"], "атака" if r["label"] else "справжнє") for r in rows)
    print(f"записано {len(rows)} викликів у {args.out}")
    for (subj, kind), n in sorted(by.items()):
        print(f"  {subj}: {kind} – {n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

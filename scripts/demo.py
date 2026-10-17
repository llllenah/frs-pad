"""Демонстрація сеансу верифікації з вебкамерою та вимірювання FPS і затримки.

Приклади:
    python scripts/demo.py                       # вебкамера 0, 1280 × 720
    python scripts/demo.py --camera 1            # інша камера (наприклад, OBS Virtual Camera)
    python scripts/demo.py --enrolled data/enrolled/P01.npy.enc   # потрібен PAD_DATASET_KEY
    python scripts/demo.py --video test.mp4 --headless --auto-start   # без вікна, з файлу

Клавіші у вікні: S – почати сеанс, E – зареєструвати обличчя заявленої особи
(30 кадрів), Q або Esc – вихід. Після виходу у runs/ зберігається звіт
demo_stats_*.json: FPS, час етапів, додаткові витрати модуля виявлення атак
на кадр і затримка рішення для кожного сеансу.

Ключ PAD_DATASET_KEY для демонстрації не обов'язковий. Якщо його задано, ембединг
з клавіші E зберігається зашифрованим у data/enrolled/demo.npy.enc. Без ключа
ембединг лишається лише в пам'яті до виходу: відкритий біометричний шаблон
на диск не пишеться (FR-65).
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pad.app import DemoApp  # noqa: E402
from pad.cli import add_common_args, load_enrolled, open_source, setup_console  # noqa: E402
from pad.runtime import build_runtime, load_config  # noqa: E402
from pad.storage import DatasetKeyError, load_key_if_set  # noqa: E402


def main() -> int:
    setup_console()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    ap.add_argument("--stats", default=None, help="куди зберегти звіт про швидкодію")
    args = ap.parse_args()

    try:
        key = load_key_if_set()
    except DatasetKeyError as exc:
        sys.exit(str(exc))
    enrolled = load_enrolled(args.enrolled, key)
    enroll_path = ROOT / "data" / "enrolled" / "demo.npy" if key is not None else None
    if key is None:
        print("PAD_DATASET_KEY не задано: ембединг з клавіші E не зберігається на диск")

    cfg = load_config(args.config)
    pipeline, manager = build_runtime(cfg, args.classifier)
    cap, file_fps = open_source(args, cfg)
    stats = Path(args.stats) if args.stats else ROOT / "runs" / f"demo_stats_{datetime.now():%Y%m%d_%H%M%S}.json"
    app = DemoApp(cap, pipeline, manager, enrolled=enrolled, headless=args.headless,
                  max_frames=args.max_frames, auto_start=args.auto_start, file_fps=file_fps,
                  enroll_path=enroll_path, dataset_key=key, stats_path=stats)
    summary = app.run()
    cap.release()
    short = {k: summary[k] for k in ("frames", "fps_wall", "fps_processing", "frame_ms",
                                     "pad_overhead_ms", "decisions")}
    print(json.dumps(short, ensure_ascii=False, indent=2))
    print(f"звіт: {stats}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

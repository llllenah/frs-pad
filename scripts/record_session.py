"""Запис сеансів пілотного набору даних (спрощена екранна форма рисунка 4.8).

Для кожного сеансу зберігається тека data/recordings/<псевдонім>_<тип>_<час>/
з файлами video.mp4.enc (кадри сеансу) і session.json.enc (анотація: псевдонім,
тип подання, інструмент підміни, цільове обличчя, умови зйомки, виклики з
часовими мітками, ряди s_ref(t), s_enr(t), ознаки, оцінка і рішення).
Обидва файли зашифровано AES-256-GCM ключем зі змінної середовища
PAD_DATASET_KEY (FR-65), відкриті video.mp4 і session.json не зберігаються.
Ембединг заявленої особи тим самим ключем зберігається в
data/enrolled/<псевдонім>.npy.enc (відкритий <псевдонім>.npy теж читається).

Запис можливий лише за наявності письмової згоди учасника (--consent).
Приклади:
    # реєстрація обличчя учасника P01 (натиснути E у вікні), потім 10 справжніх сеансів (S)
    python scripts/record_session.py --participant P01 --consent --type bona_fide --claimed P01
    # атака: Deep-Live-Cam з цільовим обличчям T01 подається через OBS Virtual Camera (камера 1)
    python scripts/record_session.py --participant P01 --consent --type attack \\
        --tool deep_live_cam --target T01 --claimed T01 --camera 1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from pad.app import DemoApp  # noqa: E402
from pad.cli import add_common_args, load_enrolled, open_source, setup_console  # noqa: E402
from pad.runtime import build_runtime, load_config  # noqa: E402
from pad.storage import (KEY_ENV, DatasetKeyError, find_embedding, load_key,  # noqa: E402
                         save_embedding)


def enroll_from_image(pipeline, path: str) -> np.ndarray:
    """Ембединг заявленої особи з фотографії (наприклад, синтетичного цільового обличчя)."""
    img = cv2.imread(path)
    if img is None:
        sys.exit(f"не вдалося прочитати {path}")
    emb = pipeline.embed_image(img)
    if emb is None:
        sys.exit(f"на {path} обличчя не виявлено")
    return emb


def main() -> int:
    setup_console()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    ap.add_argument("--participant", required=True, help="псевдонім учасника, наприклад P01")
    ap.add_argument("--consent", action="store_true", help="підтверджую наявність письмової згоди")
    ap.add_argument("--type", choices=["bona_fide", "attack"], required=True, dest="presentation_type")
    ap.add_argument("--tool", default="", help="інструмент підміни: deep_live_cam, deepfacelive")
    ap.add_argument("--target", default="", help="псевдонім цільового обличчя")
    ap.add_argument("--claimed", default=None, help="заявлена особа (псевдонім з data/enrolled)")
    ap.add_argument("--enroll-image", default=None, help="зареєструвати заявлену особу з фотографії")
    ap.add_argument("--lighting", default="normal", help="умови освітлення")
    ap.add_argument("--webcam-model", default="", help="модель вебкамери")
    ap.add_argument("--out", default=str(ROOT / "data" / "recordings"))
    args = ap.parse_args()

    if not args.consent:
        sys.exit("запис без письмової згоди учасника заборонено (додайте --consent)")
    if args.presentation_type == "attack" and not args.tool:
        sys.exit("для атаки вкажіть інструмент підміни (--tool)")
    try:
        key = load_key()
    except DatasetKeyError as exc:
        sys.exit(f"записи зберігаються лише зашифрованими: {exc}\n"
                 f"новий ключ: python scripts/decrypt_recording.py --new-key, "
                 f"потім задайте його у {KEY_ENV}")

    cfg = load_config(args.config)
    pipeline, manager = build_runtime(cfg, args.classifier)
    claimed = args.claimed or args.participant
    enroll_path = ROOT / "data" / "enrolled" / f"{claimed}.npy"
    enrolled = None
    if args.enroll_image:
        enrolled = enroll_from_image(pipeline, args.enroll_image)
        enroll_path.parent.mkdir(parents=True, exist_ok=True)
        save_embedding(enroll_path, enrolled, key)
    elif args.enrolled:
        enrolled = load_enrolled(args.enrolled, key)
    elif (found := find_embedding(enroll_path)) is not None:
        enrolled = load_enrolled(str(found), key)
    else:
        print(f"ембединг {claimed} не знайдено: натисніть E у вікні, щоб зареєструвати обличчя")

    metadata = {
        "participant": args.participant, "consent": True,
        "presentation_type": args.presentation_type,
        "swap_tool": args.tool if args.presentation_type == "attack" else "",
        "target_face": args.target, "claimed_identity": claimed,
        "conditions": {"lighting": args.lighting, "webcam": args.webcam_model,
                       "source": args.video or f"camera:{args.camera}"},
    }
    cap, file_fps = open_source(args, cfg)
    app = DemoApp(cap, pipeline, manager, enrolled=enrolled, person_id=claimed,
                  headless=args.headless, max_frames=args.max_frames, auto_start=args.auto_start,
                  file_fps=file_fps, record_dir=Path(args.out), metadata=metadata,
                  enroll_path=enroll_path, dataset_key=key)
    app.run()
    cap.release()
    return 0


if __name__ == "__main__":
    sys.exit(main())

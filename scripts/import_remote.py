"""Імпорт запису, зробленого поза прототипом: віддалений запис учасника або офлайн-атака.

Вхід – відео (webm з remote/remote_recorder.html або mp4) і анотація frs-pad-remote/1.
Кадри проходять конвеєр прототипу (MTCNN, Face Mesh, Hands, MobileFaceNet),
виклики перевіряються тими самими правилами, що й у живому сеансі, і для
кожного сеансу анотації створюється session.json.enc у форматі record_session.py.

Структура результату (усе зашифровано AES-256-GCM ключем PAD_DATASET_KEY):
    data/recordings/<псевдонім>_<тип>_<час>/video.<ext>.enc, annotation.json.enc
    data/recordings/<псевдонім>_<тип>_<час>/<псевдонім>_<тип>_<час>_s01/session.json.enc ...
Для справжнього подання з вікна реєстрації (enroll) зберігається еталон
учасника data/enrolled/<псевдонім>.npy.enc – він потрібен як заявлена особа для атак.

Приклади:
    python scripts/import_remote.py P02_2026-11-01.webm P02_2026-11-01.json --delete-source
    python scripts/import_remote.py P02_swap_P03.mp4 P02_2026-11-01.json --type attack \\
        --tool deep_live_cam --target P03 --delete-source
    python scripts/import_remote.py P02_as_T01.mp4 P02_2026-11-01.json --type attack \
        --tool deep_live_cam --target T01 --target-image data/targets/T01.jpg --delete-source
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pad.cli import setup_console  # noqa: E402
from pad.offline import (AnnotationError, enrolled_from_video, face_ratio, run_pipeline,  # noqa: E402
                         session_records, validate_annotation)
from pad.runtime import build_runtime, load_config  # noqa: E402
from pad.settings import PadSettings  # noqa: E402
from pad.storage import (KEY_ENV, DatasetKeyError, DecryptionError, find_embedding,  # noqa: E402
                         load_embedding, load_key, save_embedding, write_encrypted)

MIN_FACE_RATIO = 0.95


def import_recording(video: Path, annotation: dict, *, presentation_type: str = "bona_fide",
                     tool: str = "", target: str = "", target_image: Path | None = None, out: Path,
                     enrolled_dir: Path, key: bytes, config: str | None = None, force: bool = False,
                     log=print) -> Path:
    """Обробляє запис і зберігає його зашифрованим; повертає теку запису."""
    validate_annotation(annotation)
    participant = annotation["participant"]
    if presentation_type == "attack" and not (tool and target):
        raise ValueError("для атаки потрібні --tool і --target")
    cfg = load_config(config)
    settings = PadSettings.from_dict(cfg.get("pad", {}))
    pipeline, _ = build_runtime(cfg)
    t0 = time.perf_counter()
    vf = run_pipeline(video, pipeline, annotation,
                      progress=lambda i, n: log(f"  кадр {i}/{n or '?'}") if i % 300 == 0 else None)
    log(f"оброблено {len(vf.frames)} кадрів {vf.size[0]}×{vf.size[1]} за {time.perf_counter() - t0:.0f} с")

    ratio = face_ratio(vf, annotation)
    log(f"контроль якості: обличчя на {ratio:.1%} кадрів поза викликами з перекриттям")
    if ratio < MIN_FACE_RATIO and not force:
        raise ValueError(f"обличчя знайдено лише на {ratio:.1%} кадрів (потрібно ≥ {MIN_FACE_RATIO:.0%}); "
                         "попросіть учасника перезаписати або додайте --force")

    if presentation_type == "bona_fide":
        claimed = participant
        enrolled = enrolled_from_video(vf, annotation, settings.min_det_score)
        if enrolled is None:
            raise ValueError("у вікні реєстрації (enroll) немає кадрів з упевненою детекцією обличчя")
        emb_path = enrolled_dir / f"{participant}.npy"
        if find_embedding(emb_path) is None:
            enrolled_dir.mkdir(parents=True, exist_ok=True)
            save_embedding(emb_path, enrolled, key)
            log(f"еталон учасника збережено: {emb_path}.enc")
    else:
        claimed = target
        found = find_embedding(enrolled_dir / f"{target}.npy")
        if found is not None:
            enrolled = load_embedding(found, key)
        elif target_image is not None:
            # синтетичне цільове обличчя: еталон з фото, як --enroll-image у record_session.py
            img = cv2.imread(str(target_image))
            enrolled = pipeline.embed_image(img) if img is not None else None
            if enrolled is None:
                raise ValueError(f"на {target_image} обличчя не виявлено")
            enrolled_dir.mkdir(parents=True, exist_ok=True)
            save_embedding(enrolled_dir / f"{target}.npy", enrolled, key)
            log(f"еталон цільової особи збережено: {enrolled_dir / target}.npy.enc")
        else:
            raise ValueError(f"немає еталона цільової особи {target}: імпортуйте її справжній запис "
                             "або вкажіть --target-image")

    stamp = time.strftime("%Y%m%d_%H%M%S")
    name = f"{participant}_{presentation_type}_{stamp}"
    rec_dir = out / name
    rec_dir.mkdir(parents=True, exist_ok=False)
    metadata = {
        "participant": participant, "consent": True, "presentation_type": presentation_type,
        "swap_tool": tool if presentation_type == "attack" else "", "target_face": target,
        "claimed_identity": claimed, "conditions": annotation.get("conditions", {}),
        "source": "remote_browser" if presentation_type == "bona_fide" else "offline_swap",
        "video_file": f"video{video.suffix}",
    }
    records = session_records(vf, annotation, settings, enrolled, metadata)
    performed = 0
    for i, rec in enumerate(records, 1):
        sub = rec_dir / f"{name}_s{i:02d}"
        sub.mkdir()
        write_encrypted(sub / "session.json", json.dumps(rec, ensure_ascii=False).encode("utf-8"), key)
        performed += sum(1 for ch in rec["decision"]["explanation"]["attempts"][0] if ch["performed"])
    total = sum(len(s.get("challenges") or []) for s in annotation["sessions"])
    write_encrypted(rec_dir / f"video{video.suffix}", video.read_bytes(), key)
    write_encrypted(rec_dir / "annotation.json", json.dumps(annotation, ensure_ascii=False).encode("utf-8"), key)
    log(f"сеансів: {len(records)}, виконано викликів: {performed} з {total}")
    log(f"записано зашифрованим: {rec_dir}")
    return rec_dir


def main() -> int:
    setup_console()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video", help="відео запису (.webm або .mp4)")
    ap.add_argument("annotation", help="анотація frs-pad-remote/1 (.json)")
    ap.add_argument("--type", choices=["bona_fide", "attack"], default="bona_fide", dest="presentation_type")
    ap.add_argument("--tool", default="", help="інструмент підміни для атаки: deep_live_cam, deepfacelive")
    ap.add_argument("--target", default="", help="псевдонім цільового обличчя атаки (заявлена особа)")
    ap.add_argument("--target-image", default=None,
                    help="фото цільової особи, якщо її еталона ще немає (синтетичне обличчя T01)")
    ap.add_argument("--out", default=str(ROOT / "data" / "recordings"))
    ap.add_argument("--enrolled-dir", default=str(ROOT / "data" / "enrolled"))
    ap.add_argument("--config", default=None)
    ap.add_argument("--force", action="store_true", help="імпортувати попри непройдений контроль якості")
    ap.add_argument("--delete-source", action="store_true",
                    help="після успішного імпорту видалити відкриті файли відео й анотації")
    args = ap.parse_args()
    try:
        key = load_key()
    except DatasetKeyError as exc:
        sys.exit(f"записи зберігаються лише зашифрованими: {exc}\n"
                 f"новий ключ: python scripts/decrypt_recording.py --new-key, потім задайте його у {KEY_ENV}")
    video, ann_path = Path(args.video), Path(args.annotation)
    try:
        annotation = json.loads(ann_path.read_text(encoding="utf-8"))
        import_recording(video, annotation, presentation_type=args.presentation_type, tool=args.tool,
                         target=args.target,
                         target_image=Path(args.target_image) if args.target_image else None,
                         out=Path(args.out), enrolled_dir=Path(args.enrolled_dir),
                         key=key, config=args.config, force=args.force)
    except (AnnotationError, ValueError, FileNotFoundError, DecryptionError) as exc:
        sys.exit(f"імпорт не виконано: {exc}")
    if args.delete_source:
        # звичайне видалення: на диску можуть лишитися залишки (див. «Обмеження прототипу» в README)
        video.unlink()
        ann_path.unlink()
        print("відкриті файли видалено")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Архів коду для Google Colab: пакет pad, генератор атак і моделі MTCNN (≈ 2 МБ).

Моделі inswapper_128 і w600k_r50 (≈ 730 МБ) у Colab завантажуються напряму
з Hugging Face, тому в архів не входять. Відео учасників в архів не входять –
їх завантажують у Colab окремо і видаляють після обробки (див. notebooks/).

    python scripts/pack_colab.py            # → dist/frs-pad-colab.zip
"""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pad.cli import setup_console  # noqa: E402


def main() -> int:
    setup_console()
    out = ROOT / "dist" / "frs-pad-colab.zip"
    out.parent.mkdir(exist_ok=True)
    files = [p for p in (ROOT / "pad").glob("*.py")]
    files += [ROOT / "scripts" / "generate_attacks.py"]
    files += sorted((ROOT / "models").glob("mtcnn_*.onnx"))
    missing = [p for p in files if not p.exists()]
    if not (ROOT / "models" / "mtcnn_pnet.onnx").exists() or missing:
        sys.exit("немає моделей MTCNN: спершу python scripts/export_models.py")
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in files:
            zf.write(p, p.relative_to(ROOT).as_posix())
    print(f"{out} ({out.stat().st_size / 1e6:.1f} МБ, файлів: {len(files)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

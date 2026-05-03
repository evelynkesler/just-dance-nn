"""
Descarga el modelo de MediaPipe Pose Landmarker.

Uso:
    uv run python scripts/download_model.py
    uv run python scripts/download_model.py --complexity full
"""

from __future__ import annotations

import argparse
from pathlib import Path
import urllib.request

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"


def download(complexity: str) -> None:
    filename = f"pose_landmarker_{complexity}.task"
    url = (
        f"https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
        f"pose_landmarker_{complexity}/float16/latest/{filename}"
    )
    out = MODELS_DIR / filename
    if out.exists():
        print(f"Ya existe: {out}")
        return
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Descargando {filename} desde Google Storage...")

    def hook(blocks: int, block_size: int, total_size: int) -> None:
        if total_size > 0:
            pct = min(100, blocks * block_size * 100 // total_size)
            print(f"\r  {pct:3d}%", end="", flush=True)

    urllib.request.urlretrieve(url, out, reporthook=hook)
    print(f"\nGuardado en: {out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--complexity",
        choices=["lite", "full", "heavy"],
        default="lite",
        help="lite: rapido. full: balance. heavy: mas preciso, mas lento.",
    )
    args = parser.parse_args()
    download(args.complexity)


if __name__ == "__main__":
    main()

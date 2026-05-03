"""
Reproduce el video original con el esqueleto extraido superpuesto, para
verificar visualmente que se trackeo al bailarin correcto.

Uso:
    uv run python src/just_dance_nn/verify_reference.py "Video 1 - Happy.mp4" \
        --reference references/"Video 1 - Happy.npz" --start 36

Controles:
    q     - salir
    espacio - pausa
    flechas izq/der - frame anterior / siguiente (cuando esta pausado)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from just_dance_nn.pose_compare import NUM_LANDMARKS

POSE_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (27, 29), (29, 31), (27, 31),
    (24, 26), (26, 28), (28, 30), (30, 32), (28, 32),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verificacion visual de la referencia extraida")
    parser.add_argument("video", type=Path)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--start", type=float, default=0.0)
    return parser.parse_args()


def draw_pose(img: np.ndarray, pose: np.ndarray) -> None:
    h, w = img.shape[:2]
    points: list[tuple[int, int] | None] = []
    for i in range(NUM_LANDMARKS):
        if pose[i, 3] > 0.3:
            points.append((int(pose[i, 0] * w), int(pose[i, 1] * h)))
        else:
            points.append(None)
    for a, b in POSE_CONNECTIONS:
        if points[a] is not None and points[b] is not None:
            cv2.line(img, points[a], points[b], (0, 255, 0), 3)
    for p in points:
        if p is not None:
            cv2.circle(img, p, 5, (0, 0, 255), -1)


def main() -> None:
    args = parse_args()
    data = np.load(args.reference, allow_pickle=True)
    landmarks = data["landmarks"]
    fps = float(data["fps"])
    n = int(data["frame_count"])
    print(f"Referencia: {n} frames @ {fps:.2f} fps")

    cap = cv2.VideoCapture(str(args.video))
    video_fps = cap.get(cv2.CAP_PROP_FPS) or fps
    start_frame = int(args.start * video_fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    idx = 0
    paused = False
    delay = max(1, int(1000 / fps))

    while True:
        if not paused:
            ok, frame = cap.read()
            if not ok or idx >= n:
                break
            draw_pose(frame, landmarks[idx])
            cv2.putText(frame, f"Frame {idx + 1}/{n}", (15, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
            cv2.imshow("Verificacion referencia", frame)
            idx += 1

        key = cv2.waitKey(delay if not paused else 0) & 0xFF
        if key == ord("q"):
            break
        elif key == ord(" "):
            paused = not paused
        elif paused and key == 83:  # flecha derecha
            idx = min(idx + 1, n - 1)
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame + idx)
            ok, frame = cap.read()
            if ok:
                draw_pose(frame, landmarks[idx])
                cv2.putText(frame, f"Frame {idx + 1}/{n}", (15, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
                cv2.imshow("Verificacion referencia", frame)
        elif paused and key == 81:  # flecha izquierda
            idx = max(idx - 1, 0)
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame + idx)
            ok, frame = cap.read()
            if ok:
                draw_pose(frame, landmarks[idx])
                cv2.putText(frame, f"Frame {idx + 1}/{n}", (15, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
                cv2.imshow("Verificacion referencia", frame)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

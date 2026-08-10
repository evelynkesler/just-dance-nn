"""
PoC: deteccion de pose en tiempo real + webcam, con detector intercambiable.

Sirve para verificar que el detector funciona y ver el esqueleto en vivo.
Corre igual en la Mac (MediaPipe, CPU) y en la Jetson (PoseNet, GPU):

    uv run python src/just_dance_nn/pose_poc.py                       # MediaPipe (default)
    uv run python src/just_dance_nn/pose_poc.py --players 2           # 2 jugadores
    python src/just_dance_nn/pose_poc.py --backend posenet            # en la Jetson

Controles en vivo:
    1 - cambiar a modo 1 jugador
    2 - cambiar a modo 2 jugadores
    q - salir
"""

import argparse
import sys
import time

import cv2
import numpy as np

from just_dance_nn.pose_backends import create_detector
from just_dance_nn.pose_compare import POSE_CONNECTIONS, VIS_IDX

MAX_PLAYERS = 4

# Paleta de colores BGR para diferenciar jugadores
PLAYER_COLORS = [
    ((0, 255, 0), (0, 0, 255)),      # P1: lineas verdes, puntos rojos
    ((255, 200, 0), (0, 200, 255)),  # P2: lineas cyan-azul, puntos naranja
    ((255, 0, 255), (0, 255, 255)),  # P3: lineas magenta, puntos amarillo
    ((0, 165, 255), (255, 255, 255)),# P4: lineas naranja, puntos blanco
]


def draw_pose(frame: np.ndarray, pose: np.ndarray, player_idx: int = 0) -> None:
    """Dibuja el esqueleto canonico (13 keypoints) con color por jugador."""
    h, w, _ = frame.shape
    line_color, point_color = PLAYER_COLORS[player_idx % len(PLAYER_COLORS)]

    pts = []
    for i in range(len(pose)):
        if pose[i, VIS_IDX] > 0.3:
            pts.append((int(pose[i, 0] * w), int(pose[i, 1] * h)))
        else:
            pts.append(None)

    for a, b in POSE_CONNECTIONS:
        if pts[a] is not None and pts[b] is not None:
            cv2.line(frame, pts[a], pts[b], line_color, 2)
    for p in pts:
        if p is not None:
            cv2.circle(frame, p, 4, point_color, -1)

    # Etiqueta del jugador sobre la cabeza (keypoint 0 = nose)
    if pts[0] is not None:
        nose_x, nose_y = pts[0]
        cv2.putText(frame, f"P{player_idx + 1}", (nose_x - 15, nose_y - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, line_color, 2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PoC de pose detection (Just Dance NN)")
    parser.add_argument(
        "--players", type=int, default=1, choices=range(1, MAX_PLAYERS + 1),
        help=f"Cantidad de jugadores a detectar (1-{MAX_PLAYERS}). Default: 1",
    )
    parser.add_argument("--backend", choices=["mediapipe", "posenet"], default="mediapipe",
                        help="Detector: mediapipe (CPU) o posenet (GPU/Jetson)")
    parser.add_argument("--cam-index", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    num_players = args.players

    detector = create_detector(args.backend, num_poses=num_players)

    cap = (
        cv2.VideoCapture(args.cam_index, cv2.CAP_DSHOW)
        if sys.platform == "win32"
        else cv2.VideoCapture(args.cam_index)
    )
    if not cap.isOpened():
        print("Error: no se pudo abrir la camara")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print(f"Pose detection iniciado con {num_players} jugador(es). Backend: {args.backend}")
    print("Controles: [1] 1 jugador  [2] 2 jugadores  [q] salir")

    fps_t0 = time.time()
    fps_count = 0
    fps_display = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)
        ts_ms = int(time.time() * 1000)
        poses = detector.detect(frame, ts_ms)

        info_lines = []
        if poses:
            info_lines.append(f"Personas detectadas: {len(poses)}")
            for idx, pose in enumerate(poses):
                draw_pose(frame, pose, player_idx=idx)
                visibility = float(np.mean(pose[:, VIS_IDX]))
                info_lines.append(f"P{idx + 1} visibilidad: {visibility:.2f}")
        else:
            info_lines.append("No se detecta persona")

        # FPS
        fps_count += 1
        if time.time() - fps_t0 >= 1.0:
            fps_display = fps_count / (time.time() - fps_t0)
            fps_count = 0
            fps_t0 = time.time()

        info_lines.insert(0, f"FPS: {fps_display:.1f}  |  Modo: {num_players} jugador(es)")

        for i, line in enumerate(info_lines):
            color = (0, 0, 255) if "No se detecta" in line else (0, 255, 0)
            cv2.putText(frame, line, (10, 30 + i * 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.imshow("Just Dance NN - Pose Detection PoC", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key in (ord("1"), ord("2")):
            new_count = 1 if key == ord("1") else 2
            if new_count != num_players:
                print(f"Cambiando a {new_count} jugador(es)...")
                detector.close()
                detector = create_detector(args.backend, num_poses=new_count)
                num_players = new_count

    cap.release()
    cv2.destroyAllWindows()
    detector.close()


if __name__ == "__main__":
    main()

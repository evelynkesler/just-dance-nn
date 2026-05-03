"""
PoC: deteccion de pose en tiempo real con MediaPipe Tasks API + webcam.

Uso:
    uv run python src/just_dance_nn/pose_poc.py                # default: 1 jugador
    uv run python src/just_dance_nn/pose_poc.py --players 2    # 2 jugadores

Controles en vivo:
    1 - cambiar a modo 1 jugador
    2 - cambiar a modo 2 jugadores
    q - salir
"""

import argparse
from pathlib import Path
import time

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

MAX_PLAYERS = 4

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "pose_landmarker_lite.task"

# Conexiones del esqueleto (indices de los 33 keypoints de MediaPipe Pose)
POSE_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),  # brazos y hombros
    (11, 23), (12, 24), (23, 24),                       # torso
    (23, 25), (25, 27), (27, 29), (29, 31), (27, 31),   # pierna izq
    (24, 26), (26, 28), (28, 30), (30, 32), (28, 32),   # pierna der
    (15, 17), (15, 19), (15, 21), (17, 19),             # mano izq
    (16, 18), (16, 20), (16, 22), (18, 20),             # mano der
    (0, 1), (1, 2), (2, 3), (3, 7),                     # cara izq
    (0, 4), (4, 5), (5, 6), (6, 8),                     # cara der
    (9, 10),                                            # boca
]


# Paleta de colores BGR para diferenciar jugadores
PLAYER_COLORS = [
    ((0, 255, 0), (0, 0, 255)),      # P1: lineas verdes, puntos rojos
    ((255, 200, 0), (0, 200, 255)),  # P2: lineas cyan-azul, puntos naranja
    ((255, 0, 255), (0, 255, 255)),  # P3: lineas magenta, puntos amarillo
    ((0, 165, 255), (255, 255, 255)),# P4: lineas naranja, puntos blanco
]


def draw_pose(frame: np.ndarray, landmarks, player_idx: int = 0) -> None:
    """Dibuja keypoints y conexiones sobre el frame con color por jugador."""
    h, w, _ = frame.shape
    points = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    line_color, point_color = PLAYER_COLORS[player_idx % len(PLAYER_COLORS)]

    for a, b in POSE_CONNECTIONS:
        if a < len(points) and b < len(points):
            cv2.line(frame, points[a], points[b], line_color, 2)

    for x, y in points:
        cv2.circle(frame, (x, y), 4, point_color, -1)

    # Etiqueta del jugador sobre la cabeza
    if points:
        nose_x, nose_y = points[0]
        cv2.putText(
            frame,
            f"P{player_idx + 1}",
            (nose_x - 15, nose_y - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            line_color,
            2,
        )


def build_landmarker(num_poses: int) -> mp_vision.PoseLandmarker:
    """Crea un PoseLandmarker configurado para `num_poses` simultaneas."""
    base_options = mp_python.BaseOptions(model_asset_path=str(MODEL_PATH))
    options = mp_vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_poses=num_poses,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return mp_vision.PoseLandmarker.create_from_options(options)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PoC de pose detection (Just Dance NN)")
    parser.add_argument(
        "--players",
        type=int,
        default=1,
        choices=range(1, MAX_PLAYERS + 1),
        help=f"Cantidad de jugadores a detectar (1-{MAX_PLAYERS}). Default: 1",
    )
    return parser.parse_args()


def main() -> None:
    if not MODEL_PATH.exists():
        print(f"Error: no se encontro el modelo en {MODEL_PATH}")
        return

    args = parse_args()
    num_players = args.players

    landmarker = build_landmarker(num_players)

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("Error: no se pudo abrir la camara")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print(f"Pose detection iniciado con {num_players} jugador(es).")
    print("Controles: [1] 1 jugador  [2] 2 jugadores  [q] salir")

    fps_t0 = time.time()
    fps_count = 0
    fps_display = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        ts_ms = int(time.time() * 1000)
        result = landmarker.detect_for_video(mp_image, ts_ms)

        info_lines: list[str] = []

        if result.pose_landmarks:
            info_lines.append(f"Personas detectadas: {len(result.pose_landmarks)}")
            for idx, landmarks in enumerate(result.pose_landmarks):
                draw_pose(frame, landmarks, player_idx=idx)
                visibility = float(np.mean([lm.visibility for lm in landmarks]))
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
            cv2.putText(
                frame,
                line,
                (10, 30 + i * 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )

        cv2.imshow("Just Dance NN - Pose Detection PoC", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key in (ord("1"), ord("2")) and not (
            (key == ord("1") and num_players == 1)
            or (key == ord("2") and num_players == 2)
        ):
            new_count = 1 if key == ord("1") else 2
            print(f"Cambiando a {new_count} jugador(es)...")
            landmarker.close()
            landmarker = build_landmarker(new_count)
            num_players = new_count

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()


if __name__ == "__main__":
    main()

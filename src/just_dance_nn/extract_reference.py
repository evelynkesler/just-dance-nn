"""
Extrae landmarks de pose de un video de referencia, trackeando UNA persona
especifica (el bailarin principal) a lo largo de todo el video.

Estrategia:
    1. Procesa el primer frame con deteccion multi-persona (num_poses=4).
    2. Muestra ese frame con cada bailarin numerado y pide al usuario que
       click sobre el principal.
    3. Para los frames siguientes, asocia cada deteccion a la persona trackeada
       por minima distancia entre centroides (centro = promedio de caderas).
    4. Si en algun frame no se detecta ningun bailarin cerca del ultimo
       conocido, se marca como perdido y se mantiene la ultima posicion.

Uso:
    uv run python src/just_dance_nn/extract_reference.py <video> [--start S] [--end E] [--preview]
"""

from __future__ import annotations

import argparse
from pathlib import Path
import time

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from just_dance_nn.pose_compare import landmarks_to_array, NUM_LANDMARKS

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "pose_landmarker_lite.task"
REFERENCES_DIR = PROJECT_ROOT / "references"

NUM_POSES_DETECT = 4   # cuantas personas detectar simultaneamente
LEFT_HIP = 23
RIGHT_HIP = 24
MAX_TRACKING_DIST = 0.20   # distancia maxima (en coords normalizadas) para asociar al tracked


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extrae landmarks de un video con tracking de bailarin principal")
    parser.add_argument("video", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--preview", action="store_true",
                        help="Muestra ventana con la pose trackeada mientras procesa")
    parser.add_argument("--model-complexity", choices=["lite", "full", "heavy"], default="lite")
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--end", type=float, default=None)
    parser.add_argument("--auto-center", action="store_true",
                        help="No pedir click: tomar la persona mas cercana al centro del frame")
    return parser.parse_args()


def get_model_path(complexity: str) -> Path:
    filename = f"pose_landmarker_{complexity}.task"
    path = MODEL_PATH.parent / filename
    if not path.exists():
        url = (
            f"https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
            f"pose_landmarker_{complexity}/float16/latest/{filename}"
        )
        print(f"Modelo {complexity} no encontrado. Descargalo desde:\n  {url}")
        print(f"Y guardalo en: {path}")
        raise SystemExit(1)
    return path


def pose_centroid(pose: np.ndarray) -> tuple[float, float]:
    """Centroide del bailarin = promedio de caderas (puntos estables)."""
    return (
        float((pose[LEFT_HIP, 0] + pose[RIGHT_HIP, 0]) / 2),
        float((pose[LEFT_HIP, 1] + pose[RIGHT_HIP, 1]) / 2),
    )


def select_target_interactive(frame: np.ndarray, poses: list[np.ndarray]) -> int:
    """Muestra el frame con cada pose numerada y devuelve el indice elegido por click."""
    h, w = frame.shape[:2]
    overlay = frame.copy()
    centroids: list[tuple[int, int]] = []
    colors = [(0, 255, 0), (0, 165, 255), (255, 0, 255), (255, 255, 0)]
    for i, pose in enumerate(poses):
        color = colors[i % len(colors)]
        # esqueleto basico
        for a, b in [(11, 12), (11, 23), (12, 24), (23, 24),
                     (11, 13), (13, 15), (12, 14), (14, 16),
                     (23, 25), (25, 27), (24, 26), (26, 28)]:
            if pose[a, 3] > 0.3 and pose[b, 3] > 0.3:
                pa = (int(pose[a, 0] * w), int(pose[a, 1] * h))
                pb = (int(pose[b, 0] * w), int(pose[b, 1] * h))
                cv2.line(overlay, pa, pb, color, 3)
        cx, cy = pose_centroid(pose)
        ix, iy = int(cx * w), int(cy * h)
        centroids.append((ix, iy))
        cv2.circle(overlay, (ix, iy), 18, color, -1)
        cv2.putText(overlay, str(i + 1), (ix - 8, iy + 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 3)

    cv2.putText(overlay, "Click sobre el bailarin principal (o tecla 1-4)",
                (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(overlay, "ESC para usar el de mas al centro",
                (15, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    selected: dict[str, int] = {}

    def on_mouse(event, x, y, _flags, _param) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            dists = [((x - cx) ** 2 + (y - cy) ** 2) ** 0.5 for cx, cy in centroids]
            selected["idx"] = int(np.argmin(dists))

    win = "Selecciona bailarin"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.imshow(win, overlay)
    cv2.setMouseCallback(win, on_mouse)

    while "idx" not in selected:
        key = cv2.waitKey(20) & 0xFF
        if key in (ord("1"), ord("2"), ord("3"), ord("4")):
            n = key - ord("0")
            if n <= len(poses):
                selected["idx"] = n - 1
                break
        elif key == 27:  # ESC: pick centermost
            cx_target, cy_target = w // 2, h // 2
            dists = [((cx - cx_target) ** 2 + (cy - cy_target) ** 2) ** 0.5 for cx, cy in centroids]
            selected["idx"] = int(np.argmin(dists))
            break

    cv2.destroyWindow(win)
    return selected["idx"]


def find_first_frame_with_poses(cap, landmarker, fps, start_frame, end_frame, max_attempts=120):
    """Busca el primer frame donde se detecten poses. Devuelve (frame_idx, frame, poses)."""
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    for offset in range(max_attempts):
        ok, frame = cap.read()
        if not ok:
            return None
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts_ms = int(((start_frame + offset) / fps) * 1000)
        result = landmarker.detect_for_video(mp_image, ts_ms)
        if result.pose_landmarks:
            poses = [landmarks_to_array(lms) for lms in result.pose_landmarks]
            return offset, frame, poses
    return None


def extract(video_path: Path, output_path: Path, preview: bool, complexity: str,
            start_s: float, end_s: float | None, auto_center: bool) -> None:
    if not video_path.exists():
        raise SystemExit(f"Error: el video no existe: {video_path}")
    model_path = get_model_path(complexity)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"Error: no se pudo abrir {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    start_frame = max(0, int(start_s * fps))
    end_frame = total_frames if end_s is None else min(total_frames, int(end_s * fps))
    frame_count = end_frame - start_frame
    duration = frame_count / fps if fps > 0 else 0.0
    if frame_count <= 0:
        raise SystemExit("Rango de frames invalido")

    print(f"Video: {video_path.name}  ({width}x{height} @ {fps:.2f}fps)")
    print(f"Procesando {frame_count} frames ({duration:.1f}s) con num_poses={NUM_POSES_DETECT}")
    print(f"Modelo: {complexity}")

    base_options = mp_python.BaseOptions(model_asset_path=str(model_path))
    options = mp_vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_poses=NUM_POSES_DETECT,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    landmarker = mp_vision.PoseLandmarker.create_from_options(options)

    # Buscar el primer frame con detecciones para hacer la seleccion
    first = find_first_frame_with_poses(cap, landmarker, fps, start_frame, end_frame)
    if first is None:
        raise SystemExit("No se detectaron personas en los primeros frames del rango")
    init_offset, init_frame, init_poses = first
    print(f"Personas detectadas en el frame inicial: {len(init_poses)}")

    if auto_center or len(init_poses) == 1:
        cx_t, cy_t = 0.5, 0.5
        dists = [((c[0] - cx_t) ** 2 + (c[1] - cy_t) ** 2) ** 0.5 for c in [pose_centroid(p) for p in init_poses]]
        target_idx = int(np.argmin(dists))
        print(f"Bailarin principal seleccionado automaticamente: #{target_idx + 1} (mas central)")
    else:
        target_idx = select_target_interactive(init_frame, init_poses)
        print(f"Bailarin principal seleccionado por usuario: #{target_idx + 1}")

    last_centroid = pose_centroid(init_poses[target_idx])

    # Reset y procesar todo el rango (incluyendo los frames previos al init_offset)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    all_landmarks = np.zeros((frame_count, NUM_LANDMARKS, 4), dtype=np.float32)
    detected_count = 0
    lost_count = 0
    frame_idx = 0
    t_start = time.time()

    # Recordamos las primeras detecciones que ya hicimos para no repetir
    cached_first: list[list[np.ndarray] | None] = [None] * (init_offset + 1)
    cached_first[init_offset] = init_poses

    while frame_idx < frame_count:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_idx <= init_offset and cached_first[frame_idx] is not None:
            poses = cached_first[frame_idx]  # type: ignore
        else:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            ts_ms = int(((start_frame + frame_idx) / fps) * 1000)
            result = landmarker.detect_for_video(mp_image, ts_ms)
            poses = [landmarks_to_array(lms) for lms in result.pose_landmarks] if result.pose_landmarks else []

        chosen: np.ndarray | None = None
        if poses:
            centroids = [pose_centroid(p) for p in poses]
            dists = [((c[0] - last_centroid[0]) ** 2 + (c[1] - last_centroid[1]) ** 2) ** 0.5 for c in centroids]
            best = int(np.argmin(dists))
            if dists[best] <= MAX_TRACKING_DIST:
                chosen = poses[best]
                last_centroid = centroids[best]
                all_landmarks[frame_idx] = chosen
                detected_count += 1
            else:
                lost_count += 1
        else:
            lost_count += 1

        if preview:
            disp = frame.copy()
            for i, pose in enumerate(poses):
                color = (0, 255, 0) if (chosen is not None and pose is chosen) else (80, 80, 80)
                for a, b in [(11, 12), (11, 23), (12, 24), (23, 24),
                             (11, 13), (13, 15), (12, 14), (14, 16),
                             (23, 25), (25, 27), (24, 26), (26, 28)]:
                    if pose[a, 3] > 0.3 and pose[b, 3] > 0.3:
                        pa = (int(pose[a, 0] * width), int(pose[a, 1] * height))
                        pb = (int(pose[b, 0] * width), int(pose[b, 1] * height))
                        cv2.line(disp, pa, pb, color, 2)
            cv2.putText(disp, f"Frame {frame_idx + 1}/{frame_count}  trackeado: {detected_count}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.imshow("Extraccion (verde=trackeado)", disp)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("Cancelado por el usuario.")
                break

        frame_idx += 1
        if frame_idx % 60 == 0 or frame_idx == frame_count:
            pct = 100 * frame_idx / frame_count
            elapsed = time.time() - t_start
            eta = (elapsed / frame_idx) * (frame_count - frame_idx) if frame_idx else 0
            print(f"  {frame_idx}/{frame_count} ({pct:.1f}%)  trackeado={detected_count}  perdido={lost_count}  ETA {eta:.1f}s")

    cap.release()
    landmarker.close()
    if preview:
        cv2.destroyAllWindows()

    elapsed = time.time() - t_start
    rate = 100 * detected_count / frame_idx if frame_idx else 0
    print(f"\nProcesado en {elapsed:.1f}s")
    print(f"Bailarin principal trackeado en: {detected_count}/{frame_idx} frames ({rate:.1f}%)")
    if lost_count:
        print(f"Frames perdidos: {lost_count}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        landmarks=all_landmarks[:frame_idx],
        fps=np.float32(fps),
        duration=np.float32(duration),
        frame_count=np.int32(frame_idx),
        source=str(video_path.name),
    )
    print(f"Referencia guardada en: {output_path}")


def main() -> None:
    args = parse_args()
    output = args.output or REFERENCES_DIR / f"{args.video.stem}.npz"
    extract(args.video, output, args.preview, args.model_complexity,
            args.start, args.end, args.auto_center)


if __name__ == "__main__":
    main()

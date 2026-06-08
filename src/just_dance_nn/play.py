"""
Motor de juego: reproduce el video de referencia sincronizado con la webcam,
detecta la pose del jugador en tiempo real y calcula score por similitud.

Uso:
    uv run python src/just_dance_nn/play.py "Video 1 - Happy.mp4" \
        --reference "references/Video 1 - Happy.npz" --start 36 --end 207

Controles:
    q - salir antes de tiempo
    espacio - pausar / reanudar
"""

from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path
import sys
import time

import cv2
import numpy as np

from just_dance_nn.pose_backends import create_detector
from just_dance_nn.pose_compare import (
    NUM_KEYPOINTS,
    POSE_CONNECTIONS,
    VIS_IDX,
    compare_poses,
    mirror_pose,
    score_to_label,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Layout
PANEL_W = 640
PANEL_H = 720
HUD_H = 220
WINDOW_W = PANEL_W * 2
WINDOW_H = PANEL_H + HUD_H

# Paleta
BG_DARK = (18, 14, 30)
BG_PANEL = (28, 22, 45)
BG_PANEL_LIGHT = (40, 32, 60)
ACCENT_PINK = (180, 80, 255)
ACCENT_CYAN = (255, 220, 80)
TEXT_PRIMARY = (240, 240, 250)
TEXT_DIM = (160, 160, 180)

LABEL_COLORS = {
    "PERFECT": (255, 240, 100),   # cyan-amarillo brillante
    "GREAT":   (120, 255, 120),
    "GOOD":    (80, 200, 255),
    "OK":      (80, 165, 255),
    "MISS":    (90, 90, 255),
}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Motor Just Dance NN")
    parser.add_argument("video", type=Path)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--end", type=float, default=None)
    parser.add_argument("--cam-index", type=int, default=0)
    parser.add_argument("--countdown", type=int, default=3)
    parser.add_argument("--no-mirror", action="store_true")
    parser.add_argument("--backend", choices=["mediapipe", "posenet"], default="mediapipe",
                        help="Detector de pose: mediapipe (CPU, Mac/PC) o posenet (GPU, Jetson)")
    parser.add_argument("--model-complexity", choices=["lite", "full", "heavy"], default="lite",
                        help="Solo para backend mediapipe")
    return parser.parse_args()


# -------------------------- helpers de dibujo --------------------------

def fit_to_panel(img: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    h, w = img.shape[:2]
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (new_w, new_h))
    canvas = np.full((target_h, target_w, 3), BG_DARK, dtype=np.uint8)
    x = (target_w - new_w) // 2
    y = (target_h - new_h) // 2
    canvas[y : y + new_h, x : x + new_w] = resized
    return canvas


def rounded_rect(img: np.ndarray, p1: tuple[int, int], p2: tuple[int, int],
                 color, radius: int = 12, thickness: int = -1) -> None:
    """Rectangulo con esquinas redondeadas."""
    x1, y1 = p1
    x2, y2 = p2
    if thickness < 0:
        cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2), color, -1)
        cv2.rectangle(img, (x1, y1 + radius), (x2, y2 - radius), color, -1)
        cv2.circle(img, (x1 + radius, y1 + radius), radius, color, -1)
        cv2.circle(img, (x2 - radius, y1 + radius), radius, color, -1)
        cv2.circle(img, (x1 + radius, y2 - radius), radius, color, -1)
        cv2.circle(img, (x2 - radius, y2 - radius), radius, color, -1)
    else:
        cv2.line(img, (x1 + radius, y1), (x2 - radius, y1), color, thickness)
        cv2.line(img, (x1 + radius, y2), (x2 - radius, y2), color, thickness)
        cv2.line(img, (x1, y1 + radius), (x1, y2 - radius), color, thickness)
        cv2.line(img, (x2, y1 + radius), (x2, y2 - radius), color, thickness)
        cv2.ellipse(img, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, thickness)
        cv2.ellipse(img, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color, thickness)
        cv2.ellipse(img, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, color, thickness)
        cv2.ellipse(img, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color, thickness)


def vertical_gradient(h: int, w: int, top: tuple[int, int, int], bottom: tuple[int, int, int]) -> np.ndarray:
    """Crea un fondo con gradiente vertical."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    for y in range(h):
        t = y / max(1, h - 1)
        b = int(top[0] * (1 - t) + bottom[0] * t)
        g = int(top[1] * (1 - t) + bottom[1] * t)
        r = int(top[2] * (1 - t) + bottom[2] * t)
        img[y, :] = (b, g, r)
    return img


def put_centered_text(img: np.ndarray, text: str, y: int, font_scale: float,
                      color, thickness: int, x_center: int | None = None,
                      shadow: bool = True) -> None:
    size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, font_scale, thickness)
    cx = x_center if x_center is not None else img.shape[1] // 2
    x = cx - size[0] // 2
    if shadow:
        cv2.putText(img, text, (x + 2, y + 3), cv2.FONT_HERSHEY_DUPLEX,
                    font_scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_DUPLEX,
                font_scale, color, thickness, cv2.LINE_AA)


def draw_skeleton_glow(img: np.ndarray, landmarks: np.ndarray, base_color, glow_color) -> None:
    """Esqueleto con efecto glow: capa exterior gruesa difusa + interior fina nitida."""
    h, w = img.shape[:2]
    pts: list[tuple[int, int] | None] = []
    for i in range(NUM_KEYPOINTS):
        if landmarks[i, VIS_IDX] > 0.3:
            pts.append((int(landmarks[i, 0] * w), int(landmarks[i, 1] * h)))
        else:
            pts.append(None)

    overlay = img.copy()
    for a, b in POSE_CONNECTIONS:
        if pts[a] is not None and pts[b] is not None:
            cv2.line(overlay, pts[a], pts[b], glow_color, 9, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)

    for a, b in POSE_CONNECTIONS:
        if pts[a] is not None and pts[b] is not None:
            cv2.line(img, pts[a], pts[b], base_color, 3, cv2.LINE_AA)
    for p in pts:
        if p is not None:
            cv2.circle(img, p, 5, base_color, -1, cv2.LINE_AA)
            cv2.circle(img, p, 7, glow_color, 2, cv2.LINE_AA)


def panel_label(canvas: np.ndarray, text: str, x: int, y: int, color) -> None:
    """Etiqueta tipo 'badge' en la esquina del panel de video."""
    size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 0.7, 2)
    pad = 14
    rounded_rect(canvas, (x, y), (x + size[0] + pad * 2, y + size[1] + pad * 2),
                 (0, 0, 0), radius=10, thickness=-1)
    rounded_rect(canvas, (x, y), (x + size[0] + pad * 2, y + size[1] + pad * 2),
                 color, radius=10, thickness=2)
    cv2.putText(canvas, text, (x + pad, y + size[1] + pad - 3),
                cv2.FONT_HERSHEY_DUPLEX, 0.7, color, 2, cv2.LINE_AA)


def draw_hud(canvas: np.ndarray, smoothed_score: float, avg_score: float,
             label: str, label_age: float, combo: int, max_combo: int,
             progress: float, elapsed: float, total_duration: float, fps: float) -> None:
    """HUD inferior con cajas redondeadas, gradiente y label animado."""
    hud_y0 = PANEL_H

    # Fondo HUD con gradiente
    grad = vertical_gradient(HUD_H, WINDOW_W, BG_PANEL, BG_DARK)
    canvas[hud_y0:WINDOW_W if False else hud_y0 + HUD_H, :] = grad

    # ---- Caja izquierda: SCORE ACTUAL ----
    box_pad = 25
    left_box_x = box_pad
    left_box_w = 380
    left_box_y = hud_y0 + 18
    left_box_h = 130
    rounded_rect(canvas, (left_box_x, left_box_y),
                 (left_box_x + left_box_w, left_box_y + left_box_h),
                 BG_PANEL_LIGHT, radius=14, thickness=-1)
    rounded_rect(canvas, (left_box_x, left_box_y),
                 (left_box_x + left_box_w, left_box_y + left_box_h),
                 ACCENT_PINK, radius=14, thickness=2)
    cv2.putText(canvas, "SCORE", (left_box_x + 18, left_box_y + 30),
                cv2.FONT_HERSHEY_DUPLEX, 0.7, TEXT_DIM, 1, cv2.LINE_AA)
    cv2.putText(canvas, f"{smoothed_score:5.1f}",
                (left_box_x + 18, left_box_y + 110),
                cv2.FONT_HERSHEY_DUPLEX, 2.6, ACCENT_PINK, 5, cv2.LINE_AA)

    # ---- Caja derecha: PROMEDIO ----
    right_box_w = 380
    right_box_x = WINDOW_W - box_pad - right_box_w
    rounded_rect(canvas, (right_box_x, left_box_y),
                 (right_box_x + right_box_w, left_box_y + left_box_h),
                 BG_PANEL_LIGHT, radius=14, thickness=-1)
    rounded_rect(canvas, (right_box_x, left_box_y),
                 (right_box_x + right_box_w, left_box_y + left_box_h),
                 ACCENT_CYAN, radius=14, thickness=2)
    cv2.putText(canvas, "PROMEDIO",
                (right_box_x + 18, left_box_y + 30),
                cv2.FONT_HERSHEY_DUPLEX, 0.7, TEXT_DIM, 1, cv2.LINE_AA)
    cv2.putText(canvas, f"{avg_score:5.1f}",
                (right_box_x + 18, left_box_y + 110),
                cv2.FONT_HERSHEY_DUPLEX, 2.6, ACCENT_CYAN, 5, cv2.LINE_AA)

    # ---- Combo ----
    if combo >= 3:
        combo_text = f"x{combo} COMBO"
        size, _ = cv2.getTextSize(combo_text, cv2.FONT_HERSHEY_DUPLEX, 0.9, 2)
        cx = WINDOW_W // 2
        # Pulso del combo
        pulse = 1.0 + 0.05 * np.sin(time.time() * 10)
        scale = 0.9 * pulse
        size, _ = cv2.getTextSize(combo_text, cv2.FONT_HERSHEY_DUPLEX, scale, 2)
        x = cx - size[0] // 2
        y = hud_y0 + 50
        cv2.putText(canvas, combo_text, (x + 2, y + 2),
                    cv2.FONT_HERSHEY_DUPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(canvas, combo_text, (x, y),
                    cv2.FONT_HERSHEY_DUPLEX, scale, ACCENT_PINK, 2, cv2.LINE_AA)

    # ---- Label central animado ----
    if label and label_age < 0.6:
        # Anim: scale up al inicio (0..0.15s), zoom levemente, fade en los ultimos 0.2s
        t = label_age
        if t < 0.15:
            scale = 1.5 + (t / 0.15) * 1.2  # 1.5 -> 2.7
            alpha = 1.0
        elif t < 0.4:
            scale = 2.7
            alpha = 1.0
        else:
            scale = 2.7 + (t - 0.4) * 0.5
            alpha = max(0.0, 1.0 - (t - 0.4) / 0.2)

        color = LABEL_COLORS.get(label, TEXT_PRIMARY)
        size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, scale, 6)
        cx = WINDOW_W // 2
        x = cx - size[0] // 2
        y = hud_y0 + 130

        overlay = canvas.copy()
        cv2.putText(overlay, label, (x + 3, y + 3),
                    cv2.FONT_HERSHEY_DUPLEX, scale, (0, 0, 0), 8, cv2.LINE_AA)
        cv2.putText(overlay, label, (x, y),
                    cv2.FONT_HERSHEY_DUPLEX, scale, color, 6, cv2.LINE_AA)
        cv2.addWeighted(overlay, alpha, canvas, 1 - alpha, 0, canvas)

    # ---- Barra de progreso ----
    bar_y = hud_y0 + HUD_H - 35
    bar_h = 12
    bar_x0 = box_pad
    bar_x1 = WINDOW_W - box_pad
    rounded_rect(canvas, (bar_x0, bar_y), (bar_x1, bar_y + bar_h),
                 (50, 40, 65), radius=6, thickness=-1)
    fill_x = int(bar_x0 + (bar_x1 - bar_x0) * np.clip(progress, 0, 1))
    if fill_x > bar_x0 + 12:
        rounded_rect(canvas, (bar_x0, bar_y), (fill_x, bar_y + bar_h),
                     ACCENT_PINK, radius=6, thickness=-1)

    cv2.putText(canvas,
                f"{int(elapsed):02d}:{int(elapsed * 100) % 100:02d} / "
                f"{int(total_duration):02d}:{int(total_duration * 100) % 100:02d}",
                (bar_x0, bar_y - 8), cv2.FONT_HERSHEY_DUPLEX, 0.5, TEXT_DIM, 1, cv2.LINE_AA)
    cv2.putText(canvas, f"MAX COMBO {max_combo}   |   {fps:.0f} FPS",
                (WINDOW_W - 280, bar_y - 8), cv2.FONT_HERSHEY_DUPLEX, 0.5, TEXT_DIM, 1, cv2.LINE_AA)


def show_countdown(cap_webcam: cv2.VideoCapture, seconds: int) -> None:
    t0 = time.time()
    while True:
        ok, frame = cap_webcam.read()
        if not ok:
            continue
        frame = cv2.flip(frame, 1)
        webcam_panel = fit_to_panel(frame, PANEL_W, PANEL_H)

        ref_panel = vertical_gradient(PANEL_H, PANEL_W, ACCENT_PINK, BG_DARK)
        put_centered_text(ref_panel, "PREPARATE", PANEL_H // 2 - 30, 1.8,
                          TEXT_PRIMARY, 4, x_center=PANEL_W // 2)
        put_centered_text(ref_panel, "para bailar", PANEL_H // 2 + 30, 1.0,
                          TEXT_PRIMARY, 2, x_center=PANEL_W // 2)

        canvas = np.full((WINDOW_H, WINDOW_W, 3), BG_DARK, dtype=np.uint8)
        canvas[:PANEL_H, :PANEL_W] = ref_panel
        canvas[:PANEL_H, PANEL_W:] = webcam_panel

        # HUD vacio bonito
        hud_grad = vertical_gradient(HUD_H, WINDOW_W, BG_PANEL, BG_DARK)
        canvas[PANEL_H:, :] = hud_grad
        put_centered_text(canvas, "JUST DANCE NN", PANEL_H + HUD_H // 2 + 10,
                          1.4, ACCENT_CYAN, 3)

        elapsed = time.time() - t0
        remaining = seconds - int(elapsed)
        if remaining <= 0:
            put_centered_text(canvas, "GO!", PANEL_H // 2 + 50, 8.0,
                              ACCENT_CYAN, 18, x_center=WINDOW_W // 2)
            cv2.imshow("Just Dance NN", canvas)
            cv2.waitKey(400)
            return

        # Animacion del numero (pulsa)
        frac = elapsed - int(elapsed)
        scale = 8.0 + np.sin(frac * np.pi) * 1.5
        num = str(remaining)
        size, _ = cv2.getTextSize(num, cv2.FONT_HERSHEY_DUPLEX, scale, 18)
        cx = (WINDOW_W - size[0]) // 2
        cy = (PANEL_H + size[1]) // 2
        cv2.putText(canvas, num, (cx + 4, cy + 4), cv2.FONT_HERSHEY_DUPLEX,
                    scale, (0, 0, 0), 22, cv2.LINE_AA)
        cv2.putText(canvas, num, (cx, cy), cv2.FONT_HERSHEY_DUPLEX,
                    scale, ACCENT_CYAN, 18, cv2.LINE_AA)

        cv2.imshow("Just Dance NN", canvas)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            raise SystemExit(0)


def stars_for_score(score: float) -> int:
    if score >= 90: return 5
    if score >= 75: return 4
    if score >= 60: return 3
    if score >= 40: return 2
    if score >= 20: return 1
    return 0


def draw_star(img: np.ndarray, center: tuple[int, int], radius: int,
              filled: bool, color_filled, color_empty) -> None:
    cx, cy = center
    pts = []
    for i in range(10):
        angle = -np.pi / 2 + i * np.pi / 5
        r = radius if i % 2 == 0 else radius * 0.45
        pts.append((int(cx + r * np.cos(angle)), int(cy + r * np.sin(angle))))
    arr = np.array(pts, dtype=np.int32)
    if filled:
        cv2.fillPoly(img, [arr], color_filled, cv2.LINE_AA)
        cv2.polylines(img, [arr], True, (0, 0, 0), 2, cv2.LINE_AA)
    else:
        cv2.fillPoly(img, [arr], (40, 30, 50), cv2.LINE_AA)
        cv2.polylines(img, [arr], True, color_empty, 2, cv2.LINE_AA)


def show_final_screen(total_score: float, score_history: list[float], max_combo: int) -> None:
    canvas = vertical_gradient(WINDOW_H, WINDOW_W, BG_PANEL, BG_DARK)

    put_centered_text(canvas, "RESULTADO FINAL", 90, 1.6, TEXT_PRIMARY, 3)

    final_label = score_to_label(total_score)
    label_color = LABEL_COLORS.get(final_label, TEXT_PRIMARY)

    # Score grande centrado
    put_centered_text(canvas, f"{total_score:.1f}", 280, 6.0, label_color, 12)
    put_centered_text(canvas, final_label, 360, 2.6, label_color, 5)

    # Estrellas
    n_stars = stars_for_score(total_score)
    star_y = 460
    star_radius = 36
    star_spacing = 90
    total_w = star_spacing * 5
    start_x = (WINDOW_W - total_w) // 2 + star_spacing // 2
    for i in range(5):
        cx = start_x + i * star_spacing
        draw_star(canvas, (cx, star_y), star_radius, i < n_stars,
                  ACCENT_CYAN, TEXT_DIM)

    # Stats
    stats_y = 560
    rounded_rect(canvas, (WINDOW_W // 2 - 350, stats_y - 30),
                 (WINDOW_W // 2 + 350, stats_y + 60),
                 BG_PANEL_LIGHT, radius=14, thickness=-1)
    put_centered_text(canvas, f"MAX COMBO  {max_combo}",
                      stats_y + 5, 0.9, ACCENT_PINK, 2)
    put_centered_text(canvas, f"FRAMES VALIDOS  {len(score_history)}",
                      stats_y + 40, 0.7, TEXT_DIM, 1)

    # Mini grafico de evolucion
    if score_history:
        graph_y0 = 660
        graph_h = 160
        graph_x0 = 80
        graph_x1 = WINDOW_W - 80
        rounded_rect(canvas, (graph_x0, graph_y0),
                     (graph_x1, graph_y0 + graph_h),
                     BG_PANEL_LIGHT, radius=14, thickness=-1)
        for ref, txt in [(50, "50"), (75, "75")]:
            y = int(graph_y0 + graph_h - (ref / 100) * graph_h)
            cv2.line(canvas, (graph_x0 + 8, y), (graph_x1 - 8, y),
                     (60, 50, 75), 1, cv2.LINE_AA)
            cv2.putText(canvas, txt, (graph_x0 - 30, y + 4),
                        cv2.FONT_HERSHEY_DUPLEX, 0.4, TEXT_DIM, 1, cv2.LINE_AA)
        n = len(score_history)
        prev = None
        for i, s in enumerate(score_history):
            x = int(graph_x0 + (i / max(1, n - 1)) * (graph_x1 - graph_x0))
            y = int(graph_y0 + graph_h - (s / 100) * graph_h)
            if prev is not None:
                cv2.line(canvas, prev, (x, y), ACCENT_PINK, 2, cv2.LINE_AA)
            prev = (x, y)
        cv2.putText(canvas, "Precision a lo largo del baile",
                    (graph_x0 + 10, graph_y0 - 10),
                    cv2.FONT_HERSHEY_DUPLEX, 0.55, TEXT_DIM, 1, cv2.LINE_AA)

    put_centered_text(canvas, "Presiona cualquier tecla para salir",
                      WINDOW_H - 25, 0.6, TEXT_DIM, 1)

    cv2.imshow("Just Dance NN", canvas)
    cv2.waitKey(0)


def main() -> None:
    args = parse_args()
    if not args.video.exists():
        raise SystemExit(f"No existe el video: {args.video}")
    if not args.reference.exists():
        raise SystemExit(f"No existe la referencia: {args.reference}")

    data = np.load(args.reference, allow_pickle=True)
    ref_landmarks = data["landmarks"]
    ref_fps = float(data["fps"])
    ref_frame_count = int(data["frame_count"])
    print(f"Referencia: {ref_frame_count} frames @ {ref_fps:.2f} fps")

    mirror_mode = not args.no_mirror
    if mirror_mode:
        print("Modo espejo activado.")
        ref_landmarks = np.stack([mirror_pose(p) for p in ref_landmarks])

    cap_ref = cv2.VideoCapture(str(args.video))
    if not cap_ref.isOpened():
        raise SystemExit("No se pudo abrir el video de referencia")
    video_fps = cap_ref.get(cv2.CAP_PROP_FPS) or ref_fps
    start_frame_video = int(args.start * video_fps)
    if start_frame_video > 0:
        cap_ref.set(cv2.CAP_PROP_POS_FRAMES, start_frame_video)
    total_duration = ref_frame_count / ref_fps

    cap_cam = (
        cv2.VideoCapture(args.cam_index, cv2.CAP_DSHOW)
        if sys.platform == "win32"
        else cv2.VideoCapture(args.cam_index)
    )
    if not cap_cam.isOpened():
        raise SystemExit("No se pudo abrir la webcam")
    cap_cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap_cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    detector = create_detector(args.backend, num_poses=1,
                               model_complexity=args.model_complexity)
    print(f"Detector de pose: {args.backend}")

    cv2.namedWindow("Just Dance NN", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Just Dance NN", WINDOW_W, WINDOW_H)

    if args.countdown > 0:
        show_countdown(cap_cam, args.countdown)

    score_history: list[float] = []
    score_buffer: deque[float] = deque(maxlen=10)
    paused = False
    paused_offset = 0.0
    pause_start = 0.0

    fps_t0 = time.time()
    fps_count = 0
    fps_display = 0.0

    combo = 0
    max_combo = 0
    last_label = ""
    label_t0 = time.time()

    t_start = time.time()
    print("Empieza el baile! Presiona 'q' para salir, espacio para pausar.")

    canvas = np.full((WINDOW_H, WINDOW_W, 3), BG_DARK, dtype=np.uint8)

    while True:
        if paused:
            elapsed = pause_start - t_start - paused_offset
        else:
            elapsed = time.time() - t_start - paused_offset

        ref_idx = int(elapsed * ref_fps)
        if ref_idx >= ref_frame_count:
            break

        target_video_frame = start_frame_video + ref_idx
        current_video_frame = int(cap_ref.get(cv2.CAP_PROP_POS_FRAMES))
        if target_video_frame > current_video_frame + 1:
            cap_ref.set(cv2.CAP_PROP_POS_FRAMES, target_video_frame)
        ok_ref, ref_frame = cap_ref.read()
        if not ok_ref:
            break

        ok_cam, cam_frame = cap_cam.read()
        if not ok_cam:
            break
        cam_frame = cv2.flip(cam_frame, 1)

        ts_ms = int(elapsed * 1000)
        player_poses = detector.detect(cam_frame, ts_ms)

        current_score = 0.0
        instant_label = "MISS"
        player_array = None
        if player_poses:
            player_array = player_poses[0]
            ref_array = ref_landmarks[ref_idx]
            fs = compare_poses(player_array, ref_array)
            if fs.valid:
                current_score = fs.score
                instant_label = score_to_label(current_score)
                score_buffer.append(current_score)
                score_history.append(current_score)

        smoothed_score = float(np.mean(score_buffer)) if score_buffer else 0.0
        avg_score = float(np.mean(score_history)) if score_history else 0.0

        # Combo: avanza si el label es GOOD o mejor (score >= 55)
        if instant_label in ("GOOD", "GREAT", "PERFECT") and player_poses:
            combo += 1
            max_combo = max(max_combo, combo)
        else:
            combo = 0

        # Detectar transicion de label para animar
        if instant_label != last_label and instant_label != "MISS":
            last_label = instant_label
            label_t0 = time.time()
        elif instant_label == "MISS" and last_label != "MISS" and time.time() - label_t0 > 0.6:
            last_label = instant_label
            label_t0 = time.time()

        label_age = time.time() - label_t0

        # ---- Render ----
        canvas[:] = BG_DARK

        if mirror_mode:
            ref_frame = cv2.flip(ref_frame, 1)
        ref_panel = fit_to_panel(ref_frame, PANEL_W, PANEL_H)
        canvas[:PANEL_H, :PANEL_W] = ref_panel

        cam_panel = fit_to_panel(cam_frame, PANEL_W, PANEL_H)
        if player_array is not None:
            # Color del esqueleto segun el score actual
            color = LABEL_COLORS.get(instant_label, (200, 200, 200))
            draw_skeleton_glow(cam_panel, player_array,
                               base_color=color, glow_color=color)
        canvas[:PANEL_H, PANEL_W:] = cam_panel

        panel_label(canvas, "REFERENCIA", 18, 18, ACCENT_CYAN)
        panel_label(canvas, "VOS", PANEL_W + 18, 18, ACCENT_PINK)

        fps_count += 1
        if time.time() - fps_t0 >= 1.0:
            fps_display = fps_count / (time.time() - fps_t0)
            fps_count = 0
            fps_t0 = time.time()

        progress = elapsed / total_duration if total_duration > 0 else 0.0
        draw_hud(canvas, smoothed_score, avg_score, last_label, label_age,
                 combo, max_combo, progress, elapsed, total_duration, fps_display)

        if paused:
            overlay = canvas.copy()
            cv2.rectangle(overlay, (0, 0), (WINDOW_W, PANEL_H), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.5, canvas, 0.5, 0, canvas)
            put_centered_text(canvas, "PAUSADO", PANEL_H // 2, 3.5,
                              ACCENT_CYAN, 8)

        cv2.imshow("Just Dance NN", canvas)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord(" "):
            if paused:
                paused_offset += time.time() - pause_start
                paused = False
            else:
                pause_start = time.time()
                paused = True

    final_score = float(np.mean(score_history)) if score_history else 0.0
    print(f"Score final: {final_score:.2f}  ({score_to_label(final_score)})  combo max: {max_combo}")
    show_final_screen(final_score, score_history, max_combo)

    cap_ref.release()
    cap_cam.release()
    detector.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

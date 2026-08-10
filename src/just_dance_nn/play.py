"""
Motor de juego: reproduce el video de referencia sincronizado con la webcam,
detecta la pose del jugador en tiempo real y calcula score por similitud.

Uso:
    uv run python src/just_dance_nn/play.py "Video 1 - Happy.mp4" \
        --reference "references/Video 1 - Happy.npz" --start 36 --end 207

Controles:
    q - salir antes de tiempo
    espacio - pausar / reanudar
    f - pantalla completa on/off
"""

import argparse
from collections import deque
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

import cv2
import numpy as np

from just_dance_nn.pose_backends import create_detector
from just_dance_nn.pose_compare import (
    NUM_KEYPOINTS,
    POSE_CONNECTIONS,
    VIS_IDX,
    best_score_in_window,
    mirror_pose,
    precompute_reference,
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
    parser.add_argument("video", type=Path, nargs="?", default=None,
                        help="Video a bailar. Si se OMITE, abre el menu de seleccion "
                             "de canciones (escanea la carpeta de videos).")
    parser.add_argument("--reference", type=Path, default=None,
                        help="Referencia .npz. Si se omite, se busca <references-dir>/<video>.npz")
    parser.add_argument("--videos-dir", type=Path, default=PROJECT_ROOT / "videos",
                        help="Carpeta donde estan los videos (para el menu)")
    parser.add_argument("--references-dir", type=Path, default=PROJECT_ROOT / "references",
                        help="Carpeta donde se guardan/buscan las referencias .npz")
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--end", type=float, default=None)
    parser.add_argument("--cam-index", type=int, default=0)
    parser.add_argument("--countdown", type=int, default=3)
    parser.add_argument("--no-mirror", action="store_true")
    parser.add_argument("--no-audio", action="store_true",
                        help="No reproducir el audio del video")
    parser.add_argument("--fullscreen", action="store_true",
                        help="Arrancar en pantalla completa (tecla F para alternar)")
    parser.add_argument("--infer-every", type=int, default=1,
                        help="Correr la red cada N frames y reusar la pose entre medio "
                             "(N=2 recomendado en la Jetson: mas fluidez, misma jugabilidad)")
    parser.add_argument("--tolerance", type=float, default=0.5,
                        help="Tolerancia temporal en segundos: se compara contra una ventana "
                             "de frames de la referencia y se toma el mejor match (compensa "
                             "el retraso natural del jugador). 0 = comparacion estricta.")
    parser.add_argument("--backend", choices=["mediapipe", "posenet"], default="mediapipe",
                        help="Detector de pose: mediapipe (CPU, Mac/PC) o posenet (GPU, Jetson)")
    parser.add_argument("--model-complexity", choices=["lite", "full", "heavy"], default="lite",
                        help="Solo para backend mediapipe")
    return parser.parse_args()


# -------------------------- audio --------------------------

class AudioPlayer:
    """Reproduce el audio del video de referencia con ffplay como subproceso.

    La sincronizacion es por reloj compartido: el proceso arranca junto con el
    juego, en el segundo `start_s` del video. La pausa congela el proceso con
    SIGSTOP y lo reanuda con SIGCONT; como el juego congela su reloj de la
    misma manera, los dos quedan alineados al reanudar.

    Si ffplay no esta instalado (p. ej. en el contenedor de la Jetson sin
    ffmpeg), el juego sigue funcionando sin sonido.
    """

    def __init__(self, video_path, start_s=0.0, enabled=True):
        self._proc = None
        self._paused = False
        self._args = self._build_cmd(video_path, start_s) if enabled else None
        if enabled and self._args is None:
            print("(sin audio: no hay ffplay, mpv ni gst-launch-1.0 disponibles. "
                  "Instalar ffmpeg para tener sonido)")

    @staticmethod
    def _build_cmd(video_path, start_s):
        """Arma el comando del reproductor con el primero disponible:
        ffplay (ffmpeg) -> mpv -> gst-launch-1.0 (GStreamer, presente en el
        contenedor jetson-inference sin instalar nada)."""
        ffplay = shutil.which("ffplay")
        if ffplay:
            return [ffplay, "-nodisp", "-autoexit", "-loglevel", "quiet",
                    "-vn", "-ss", str(start_s), "-i", str(video_path)]
        mpv = shutil.which("mpv")
        if mpv:
            return [mpv, "--no-video", "--really-quiet",
                    "--start=" + str(start_s), str(video_path)]
        gst = shutil.which("gst-launch-1.0")
        if gst and start_s < 1.0:
            # playbin no puede arrancar desde un offset por CLI, asi que solo
            # sirve cuando el juego arranca desde el principio del video (el
            # caso del menu en la Jetson, que siempre usa start=0).
            uri = Path(video_path).resolve().as_uri()
            return [gst, "-q", "playbin", "uri=" + uri, "video-sink=fakesink"]
        return None

    def start(self):
        if self._args is None:
            return
        try:
            self._proc = subprocess.Popen(
                self._args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except OSError as exc:
            print(f"(sin audio: no se pudo lanzar ffplay: {exc})")
            self._proc = None

    def _signal(self, sig):
        if self._proc is not None and self._proc.poll() is None:
            try:
                self._proc.send_signal(sig)
            except OSError:
                pass

    def pause(self):
        if not self._paused and hasattr(signal, "SIGSTOP"):
            self._signal(signal.SIGSTOP)
            self._paused = True

    def resume(self):
        if self._paused and hasattr(signal, "SIGCONT"):
            self._signal(signal.SIGCONT)
            self._paused = False

    def stop(self):
        if self._proc is not None:
            if self._paused:
                self.resume()  # un proceso frenado no atiende SIGTERM
            try:
                self._proc.terminate()
                self._proc.wait(timeout=2)
            except Exception:
                try:
                    self._proc.kill()
                except OSError:
                    pass
            self._proc = None


# -------------------------- helpers de dibujo --------------------------

def set_fullscreen(window, on):
    cv2.setWindowProperty(window, cv2.WND_PROP_FULLSCREEN,
                          cv2.WINDOW_FULLSCREEN if on else cv2.WINDOW_NORMAL)
    if not on:
        cv2.resizeWindow(window, WINDOW_W, WINDOW_H)


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


def rounded_rect(img, p1, p2, color, radius=12, thickness=-1):
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


# Cache de gradientes ya calculados. El gradiente se arma con un loop de Python
# (lento en la Jetson); si lo recalculamos en cada frame mata el rendimiento.
# Lo calculamos UNA vez por tamano/color y despues solo devolvemos una copia
# (copiar un array es una operacion rapida en C).
_grad_cache = {}


def vertical_gradient(h, w, top, bottom):
    """Crea un fondo con gradiente vertical (cacheado por tamano y colores)."""
    key = (h, w, tuple(top), tuple(bottom))
    base = _grad_cache.get(key)
    if base is None:
        base = np.zeros((h, w, 3), dtype=np.uint8)
        for y in range(h):
            t = y / max(1, h - 1)
            b = int(top[0] * (1 - t) + bottom[0] * t)
            g = int(top[1] * (1 - t) + bottom[1] * t)
            r = int(top[2] * (1 - t) + bottom[2] * t)
            base[y, :] = (b, g, r)
        _grad_cache[key] = base
    return base.copy()


def put_centered_text(img, text, y, font_scale, color, thickness,
                      x_center=None, shadow=True):
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
    pts = []
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
             progress: float, elapsed: float, total_duration: float, fps: float,
             inf_fps: float = 0.0) -> None:
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
    # render = FPS del loop completo; inf = FPS de la red neuronal sola (mide la GPU)
    cv2.putText(canvas, f"MAX COMBO {max_combo}   |   {fps:.0f} FPS   |   red {inf_fps:.0f} FPS",
                (WINDOW_W - 420, bar_y - 8), cv2.FONT_HERSHEY_DUPLEX, 0.5, TEXT_DIM, 1, cv2.LINE_AA)


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


def draw_star(img, center, radius, filled, color_filled, color_empty):
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


JOINT_GROUPS = [("CODOS", 0, 1), ("HOMBROS", 2, 3),
                ("CADERAS", 4, 5), ("RODILLAS", 6, 7)]


def show_final_screen(total_score, score_history, max_combo,
                      label_counts=None, joint_sum=None, joint_cnt=None,
                      detected_frames=0):
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
    total_frames = max(1, len(score_history))
    stats_y0 = 520
    rounded_rect(canvas, (WINDOW_W // 2 - 440, stats_y0),
                 (WINDOW_W // 2 + 440, stats_y0 + 125),
                 BG_PANEL_LIGHT, radius=14, thickness=-1)

    # Linea 1: desglose de etiquetas con colores
    if label_counts:
        parts = []
        for lbl in ("PERFECT", "GREAT", "GOOD", "OK", "MISS"):
            n = label_counts.get(lbl, 0)
            if n:
                parts.append((lbl, f"{lbl} {100 * n // total_frames}%"))
        sep = "   "
        font, fs, th = cv2.FONT_HERSHEY_DUPLEX, 0.75, 2
        widths = [cv2.getTextSize(t, font, fs, th)[0][0] for _, t in parts]
        sep_w = cv2.getTextSize(sep, font, fs, th)[0][0]
        x = WINDOW_W // 2 - (sum(widths) + sep_w * (len(parts) - 1)) // 2
        for (lbl, t), w in zip(parts, widths):
            cv2.putText(canvas, t, (x, stats_y0 + 35), font, fs,
                        LABEL_COLORS.get(lbl, TEXT_PRIMARY), th, cv2.LINE_AA)
            x += w + sep_w

    # Linea 2: precision por articulacion (que parte del cuerpo fallo)
    if joint_cnt is not None and joint_cnt.sum() > 0:
        segs = []
        for name, a, b in JOINT_GROUPS:
            cnt = joint_cnt[a] + joint_cnt[b]
            if cnt > 0:
                pct = 100.0 * (joint_sum[a] + joint_sum[b]) / cnt
                segs.append(f"{name} {pct:.0f}%")
            else:
                segs.append(f"{name} --")
        put_centered_text(canvas, "   ".join(segs), stats_y0 + 75, 0.7,
                          TEXT_PRIMARY, 1)

    # Linea 3: combo y deteccion
    det_pct = 100 * detected_frames // total_frames
    put_centered_text(canvas,
                      f"MAX COMBO {max_combo}   |   DETECTADA {det_pct}% DEL TIEMPO",
                      stats_y0 + 110, 0.65, TEXT_DIM, 1)

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

    put_centered_text(canvas, "R = bailar otra vez   |   cualquier otra tecla = salir",
                      WINDOW_H - 25, 0.6, TEXT_DIM, 1)

    cv2.imshow("Just Dance NN", canvas)
    key = cv2.waitKey(0) & 0xFF
    return key in (ord("r"), ord("R"))


def run_game(video, reference, start=0.0, end=None, cam_index=0, countdown=3,
             mirror=True, backend="mediapipe", model_complexity="lite",
             audio=True, fullscreen=False, infer_every=1, tolerance=0.5):
    """Corre una partida completa con un video y su referencia ya extraida.

    Es el motor del juego, reutilizable: lo llama tanto el modo CLI directo
    (play.py VIDEO ...) como el menu de seleccion de canciones.
    """
    video = Path(video)
    reference = Path(reference)
    if not video.exists():
        raise SystemExit(f"No existe el video: {video}")
    if not reference.exists():
        raise SystemExit(f"No existe la referencia: {reference}")

    data = np.load(reference, allow_pickle=True)
    ref_landmarks = data["landmarks"]
    ref_fps = float(data["fps"])
    ref_frame_count = int(data["frame_count"])
    print(f"Referencia: {ref_frame_count} frames @ {ref_fps:.2f} fps")

    mirror_mode = mirror
    if mirror_mode:
        print("Modo espejo activado.")
        ref_landmarks = np.stack([mirror_pose(p) for p in ref_landmarks])

    # Pre-calculo de angulos/visibilidad de toda la referencia (una vez).
    print("Preparando referencia...")
    ref_angles_all, ref_valid_all = precompute_reference(ref_landmarks)

    cap_ref = cv2.VideoCapture(str(video))
    if not cap_ref.isOpened():
        raise SystemExit("No se pudo abrir el video de referencia")
    video_fps = cap_ref.get(cv2.CAP_PROP_FPS) or ref_fps
    start_frame_video = int(start * video_fps)
    if start_frame_video > 0:
        cap_ref.set(cv2.CAP_PROP_POS_FRAMES, start_frame_video)
    total_duration = ref_frame_count / ref_fps

    cap_cam = (
        cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
        if sys.platform == "win32"
        else cv2.VideoCapture(cam_index)
    )
    if not cap_cam.isOpened():
        raise SystemExit("No se pudo abrir la webcam")
    # 640x480: suficiente para detectar la pose y MUCHO mas liviano que 720p
    # en la Jetson (acelera tanto la captura como la inferencia de la red).
    cap_cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap_cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    detector = create_detector(backend, num_poses=1,
                               model_complexity=model_complexity)
    print(f"Detector de pose: {backend}")

    cv2.namedWindow("Just Dance NN", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Just Dance NN", WINDOW_W, WINDOW_H)
    if fullscreen:
        set_fullscreen("Just Dance NN", True)

    if countdown > 0:
        show_countdown(cap_cam, countdown)

    # El audio arranca junto con el reloj del juego (mismo instante que t_start)
    audio_player = AudioPlayer(video, start_s=start, enabled=audio)
    audio_player.start()

    score_history = []
    score_buffer = deque(maxlen=10)
    paused = False
    paused_offset = 0.0
    pause_start = 0.0

    fps_t0 = time.time()
    fps_count = 0
    fps_display = 0.0

    # Medicion de la inferencia de la red por separado (la metrica que importa
    # para el TP: cuanto rinde el detector en la GPU de la Jetson).
    inf_ms_buffer = deque(maxlen=30)  # ventana para el HUD
    inf_ms_all = []                   # historico para el resumen final
    inf_fps_display = 0.0

    combo = 0
    max_combo = 0
    last_label = ""
    label_t0 = time.time()

    infer_every = max(1, int(infer_every))
    loop_i = 0
    player_poses = []   # ultima deteccion (se reusa entre inferencias)

    # Estadisticas finales: por articulacion, por etiqueta y deteccion
    joint_sum = np.zeros(8, dtype=np.float64)
    joint_cnt = np.zeros(8, dtype=np.int64)
    detected_frames = 0
    label_counts = {}

    # Tolerancia temporal: el jugador reacciona con ~200-500 ms de retraso
    # respecto a lo que ve. Comparamos contra una ventana de frames de la
    # referencia (mas hacia atras que adelante) y nos quedamos con el mejor
    # match, para puntuar el paso y no el desfase.
    tol_back = max(0, int(tolerance * ref_fps))
    tol_fwd = tol_back // 3

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
        skip = target_video_frame - current_video_frame
        if skip > 1:
            # Si esta MUY atrasado, un seek real (caro pero corta camino).
            # Si esta poco atrasado, avanzar con grab() es mucho mas barato que
            # un seek aleatorio (que reabre desde el keyframe en cada frame: ESA
            # era la causa principal del tironeo del video en la Jetson).
            if skip > 30:
                cap_ref.set(cv2.CAP_PROP_POS_FRAMES, target_video_frame)
            else:
                for _ in range(skip - 1):
                    cap_ref.grab()
        ok_ref, ref_frame = cap_ref.read()
        if not ok_ref:
            break

        ok_cam, cam_frame = cap_cam.read()
        if not ok_cam:
            break
        cam_frame = cv2.flip(cam_frame, 1)

        # Con --infer-every N la red corre 1 de cada N frames y entre medio se
        # reusa la ultima pose (la pose humana casi no cambia en ~30-60 ms).
        # En la Jetson N=2 duplica la fluidez del loop sin afectar el juego.
        if loop_i % infer_every == 0:
            ts_ms = int(elapsed * 1000)
            t_inf0 = time.perf_counter()
            player_poses = detector.detect(cam_frame, ts_ms)
            inf_ms = (time.perf_counter() - t_inf0) * 1000.0
            inf_ms_buffer.append(inf_ms)
            inf_ms_all.append(inf_ms)
        loop_i += 1

        current_score = 0.0
        instant_label = "MISS"
        player_array = None
        if player_poses:
            player_array = player_poses[0]
            lo = max(0, ref_idx - tol_back)
            hi = min(ref_frame_count - 1, ref_idx + tol_fwd)
            best = best_score_in_window(player_array,
                                        ref_angles_all[lo:hi + 1],
                                        ref_valid_all[lo:hi + 1])
            if best is not None:
                current_score, pj_scores, pj_valid = best
                instant_label = score_to_label(current_score)
                if not paused:
                    joint_sum[pj_valid] += pj_scores[pj_valid]
                    joint_cnt[pj_valid] += 1
                    detected_frames += 1

        # TODOS los frames cuentan para el promedio (0 si no hay jugador
        # detectado o la comparacion no fue confiable): no bailar = 0 puntos.
        # Durante la pausa no se acumula score.
        if not paused:
            score_buffer.append(current_score)
            score_history.append(current_score)
            label_counts[instant_label] = label_counts.get(instant_label, 0) + 1

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
            if inf_ms_buffer:
                mean_inf_ms = sum(inf_ms_buffer) / len(inf_ms_buffer)
                inf_fps_display = 1000.0 / mean_inf_ms if mean_inf_ms > 0 else 0.0
            fps_count = 0
            fps_t0 = time.time()

        progress = elapsed / total_duration if total_duration > 0 else 0.0
        draw_hud(canvas, smoothed_score, avg_score, last_label, label_age,
                 combo, max_combo, progress, elapsed, total_duration, fps_display,
                 inf_fps_display)

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
                audio_player.resume()
            else:
                pause_start = time.time()
                paused = True
                audio_player.pause()
        elif key in (ord("f"), ord("F")):
            fullscreen = not fullscreen
            set_fullscreen("Just Dance NN", fullscreen)

    audio_player.stop()

    final_score = float(np.mean(score_history)) if score_history else 0.0
    print(f"Score final: {final_score:.2f}  ({score_to_label(final_score)})  combo max: {max_combo}")

    # ---- Resumen de rendimiento de la red (para el informe del TP) ----
    if inf_ms_all:
        inf = np.array(inf_ms_all)
        mean_ms = float(inf.mean())
        p95_ms = float(np.percentile(inf, 95))
        max_ms = float(inf.max())
        print("")
        print(f"--- Rendimiento de inferencia ({backend}) ---")
        print(f"  frames procesados : {len(inf)}")
        print(f"  latencia media    : {mean_ms:.1f} ms  ->  {1000.0 / mean_ms:.1f} FPS")
        print(f"  latencia p95      : {p95_ms:.1f} ms")
        print(f"  latencia maxima   : {max_ms:.1f} ms")

    # Resumen de precision por consola (para el informe)
    if score_history:
        total_frames = len(score_history)
        print("--- Precision ---")
        print("  etiquetas: " + "  ".join(
            f"{lbl} {100 * label_counts.get(lbl, 0) // total_frames}%"
            for lbl in ("PERFECT", "GREAT", "GOOD", "OK", "MISS")))
        segs = []
        for name, a, b in JOINT_GROUPS:
            cnt = joint_cnt[a] + joint_cnt[b]
            segs.append(f"{name.lower()} {100.0 * (joint_sum[a] + joint_sum[b]) / cnt:.0f}%"
                        if cnt else f"{name.lower()} --")
        print("  por articulacion: " + " | ".join(segs))
        print(f"  detectada en camara: {100 * detected_frames // total_frames}% del tiempo")

    retry = show_final_screen(final_score, score_history, max_combo,
                              label_counts=label_counts, joint_sum=joint_sum,
                              joint_cnt=joint_cnt, detected_frames=detected_frames)

    cap_ref.release()
    cap_cam.release()
    detector.close()
    cv2.destroyAllWindows()
    return retry


# ===========================================================================
# Menu de seleccion de canciones
# ===========================================================================

VIDEO_EXTS = {".mp4", ".mkv", ".webm", ".mov", ".avi", ".m4v"}

# Teclas de navegacion (varios codigos segun SO/build de OpenCV + WASD/JK)
_UP_KEYS = {ord("w"), ord("W"), ord("k"), 82, 65362, 2490368, 63232}
_DOWN_KEYS = {ord("s"), ord("S"), ord("j"), 84, 65364, 2621440, 63233}
_ENTER_KEYS = {13, 10, ord(" ")}
_QUIT_KEYS = {ord("q"), ord("Q"), 27}


class Song:
    """Una cancion disponible: su video y (si existe) su referencia .npz.

    (Clase simple en vez de @dataclass para correr en Python 3.6 en la Jetson.)
    """

    def __init__(self, title, video, reference, has_reference):
        self.title = title
        self.video = video
        self.reference = reference
        self.has_reference = has_reference


def discover_songs(videos_dir, references_dir):
    """Escanea la carpeta de videos y arma la lista de canciones.

    Se llama en cada vuelta del menu, asi que cualquier video nuevo que se
    agregue a la carpeta aparece automaticamente como opcion. Para cada video
    busca su referencia en <references_dir>/<nombre_del_video>.npz.
    """
    videos_dir = Path(videos_dir)
    references_dir = Path(references_dir)
    songs = []
    if not videos_dir.exists():
        return songs
    for v in sorted(videos_dir.iterdir(), key=lambda p: p.name.lower()):
        if v.name.startswith(".") or not v.is_file():
            continue
        if v.suffix.lower() not in VIDEO_EXTS:
            continue
        ref = references_dir / f"{v.stem}.npz"
        songs.append(Song(title=v.stem, video=v, reference=ref,
                          has_reference=ref.exists()))
    return songs


def select_song_screen(songs, fullscreen=False):
    """Pantalla de seleccion. Devuelve la cancion elegida o None si se sale."""
    win = "Just Dance NN"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, WINDOW_W, WINDOW_H)
    if fullscreen:
        set_fullscreen(win, True)

    selected = 0
    view_start = 0
    max_visible = 7
    list_w = 960
    x0 = (WINDOW_W - list_w) // 2
    row_h = 72
    gap = 14
    top = 250

    while True:
        canvas = vertical_gradient(WINDOW_H, WINDOW_W, BG_PANEL, BG_DARK)
        put_centered_text(canvas, "JUST DANCE NN", 110, 2.0, ACCENT_CYAN, 4)
        put_centered_text(canvas, "Elegi una cancion", 175, 1.0, TEXT_PRIMARY, 2)

        if not songs:
            put_centered_text(canvas, "No hay videos en la carpeta 'videos/'",
                              WINDOW_H // 2 - 20, 1.0, TEXT_PRIMARY, 2)
            put_centered_text(canvas, "Agrega archivos .mp4 a esa carpeta y volve a entrar",
                              WINDOW_H // 2 + 30, 0.7, TEXT_DIM, 1)
            put_centered_text(canvas, "Q para salir", WINDOW_H - 40, 0.7, TEXT_DIM, 1)
            cv2.imshow(win, canvas)
            if cv2.waitKeyEx(30) in _QUIT_KEYS:
                return None
            continue

        # Mantener la seleccion dentro de la ventana visible (scroll)
        if selected < view_start:
            view_start = selected
        elif selected >= view_start + max_visible:
            view_start = selected - max_visible + 1

        visible = songs[view_start:view_start + max_visible]
        for i, song in enumerate(visible):
            real_idx = view_start + i
            y = top + i * (row_h + gap)
            is_sel = real_idx == selected
            bg = BG_PANEL_LIGHT if is_sel else BG_PANEL
            rounded_rect(canvas, (x0, y), (x0 + list_w, y + row_h), bg, radius=14, thickness=-1)
            border = ACCENT_PINK if is_sel else (60, 50, 75)
            rounded_rect(canvas, (x0, y), (x0 + list_w, y + row_h), border,
                         radius=14, thickness=2 if is_sel else 1)

            cv2.putText(canvas, f"{real_idx + 1}", (x0 + 24, y + 46),
                        cv2.FONT_HERSHEY_DUPLEX, 1.0,
                        ACCENT_CYAN if is_sel else TEXT_DIM, 2, cv2.LINE_AA)

            title = song.title if len(song.title) <= 46 else song.title[:43] + "..."
            cv2.putText(canvas, title, (x0 + 78, y + 45),
                        cv2.FONT_HERSHEY_DUPLEX, 0.75,
                        TEXT_PRIMARY if is_sel else TEXT_DIM, 1, cv2.LINE_AA)

            if song.has_reference:
                status, scol = "LISTO", (120, 255, 120)
            else:
                status, scol = "FALTA REF.", (120, 140, 255)
            ssize, _ = cv2.getTextSize(status, cv2.FONT_HERSHEY_DUPLEX, 0.6, 2)
            cv2.putText(canvas, status, (x0 + list_w - ssize[0] - 24, y + 44),
                        cv2.FONT_HERSHEY_DUPLEX, 0.6, scol, 2, cv2.LINE_AA)

        # Nota si la cancion elegida todavia no tiene referencia
        if not songs[selected].has_reference:
            put_centered_text(canvas,
                              "Esta cancion va a extraer su referencia la 1ra vez (puede tardar)",
                              WINDOW_H - 70, 0.6, ACCENT_CYAN, 1)
        put_centered_text(canvas,
                          "W/S o flechas: mover    |    ENTER: elegir    |    Q: salir",
                          WINDOW_H - 35, 0.7, TEXT_DIM, 1)

        cv2.imshow(win, canvas)
        key = cv2.waitKeyEx(30)
        if key == -1:
            continue
        if key in _QUIT_KEYS:
            return None
        if key in (ord("f"), ord("F")):
            fullscreen = not fullscreen
            set_fullscreen(win, fullscreen)
            continue
        if key in _ENTER_KEYS:
            return songs[selected]
        if key in _UP_KEYS:
            selected = (selected - 1) % len(songs)
        elif key in _DOWN_KEYS:
            selected = (selected + 1) % len(songs)
        elif ord("1") <= key <= ord("9"):
            idx = key - ord("1")
            if idx < len(songs):
                selected = idx


def main() -> None:
    args = parse_args()

    # --- Modo directo: se paso un video por linea de comandos ---
    if args.video is not None:
        reference = args.reference or (args.references_dir / f"{args.video.stem}.npz")
        while run_game(args.video, reference, start=args.start, end=args.end,
                       cam_index=args.cam_index, countdown=args.countdown,
                       mirror=not args.no_mirror, backend=args.backend,
                       model_complexity=args.model_complexity,
                       audio=not args.no_audio, fullscreen=args.fullscreen,
                       infer_every=args.infer_every, tolerance=args.tolerance):
            pass   # R en la pantalla final = bailar otra vez
        return

    # --- Modo menu: escanea la carpeta de videos en cada vuelta ---
    print(f"Menu de canciones. Carpeta de videos: {args.videos_dir}")
    while True:
        songs = discover_songs(args.videos_dir, args.references_dir)
        choice = select_song_screen(songs, fullscreen=args.fullscreen)
        if choice is None:
            print("Saliendo del menu.")
            break

        # Si la cancion no tiene referencia, se extrae sola (modo auto, sin clicks)
        if not choice.has_reference:
            print(f"'{choice.title}': extrayendo referencia por primera vez...")
            try:
                from just_dance_nn.extract_reference import extract
                extract(choice.video, choice.reference, preview=False,
                        backend=args.backend, complexity=args.model_complexity,
                        start_s=args.start, end_s=args.end, auto_center=True)
            except SystemExit as exc:
                print(f"No se pudo extraer la referencia: {exc}")
                continue

        while run_game(choice.video, choice.reference, start=args.start, end=args.end,
                       cam_index=args.cam_index, countdown=args.countdown,
                       mirror=not args.no_mirror, backend=args.backend,
                       model_complexity=args.model_complexity,
                       audio=not args.no_audio, fullscreen=args.fullscreen,
                       infer_every=args.infer_every, tolerance=args.tolerance):
            pass   # R en la pantalla final = bailar otra vez; al salir, vuelve al menu

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

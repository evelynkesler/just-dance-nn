"""
Comparador de poses basado en angulos articulares.

La idea: dos personas con distinta altura, complexion o distancia a la camara
producen coordenadas (x, y) muy distintas aun haciendo el mismo movimiento.
En cambio, los ANGULOS entre articulaciones (codo, hombro, cadera, rodilla)
son invariantes a esas diferencias y capturan la pose real.

Convencion de landmarks (MediaPipe Pose, 33 puntos):
    11 hombro_izq    12 hombro_der
    13 codo_izq      14 codo_der
    15 muneca_izq    16 muneca_der
    23 cadera_izq    24 cadera_der
    25 rodilla_izq   26 rodilla_der
    27 tobillo_izq   28 tobillo_der
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

# Articulaciones a evaluar: (nombre, punto_a, vertice, punto_b)
# El angulo es el formado en `vertice` por los segmentos vertice->a y vertice->b
JOINT_ANGLES: list[tuple[str, int, int, int]] = [
    ("codo_izq",     11, 13, 15),  # hombro - codo - muneca (izq)
    ("codo_der",     12, 14, 16),  # hombro - codo - muneca (der)
    ("hombro_izq",   13, 11, 23),  # codo - hombro - cadera (izq)
    ("hombro_der",   14, 12, 24),  # codo - hombro - cadera (der)
    ("cadera_izq",   11, 23, 25),  # hombro - cadera - rodilla (izq)
    ("cadera_der",   12, 24, 26),  # hombro - cadera - rodilla (der)
    ("rodilla_izq",  23, 25, 27),  # cadera - rodilla - tobillo (izq)
    ("rodilla_der",  24, 26, 28),  # cadera - rodilla - tobillo (der)
]

NUM_LANDMARKS = 33
NUM_ANGLES = len(JOINT_ANGLES)

# Pares izquierda/derecha de MediaPipe Pose para swappear al espejar.
# Indice 0 (nariz) y la linea media se mantienen.
LEFT_RIGHT_PAIRS: list[tuple[int, int]] = [
    (1, 4), (2, 5), (3, 6),       # ojos
    (7, 8),                        # orejas
    (9, 10),                       # boca
    (11, 12),                      # hombros
    (13, 14),                      # codos
    (15, 16),                      # munecas
    (17, 18), (19, 20), (21, 22),  # dedos / pulgares
    (23, 24),                      # caderas
    (25, 26),                      # rodillas
    (27, 28),                      # tobillos
    (29, 30),                      # talones
    (31, 32),                      # punta del pie
]


def mirror_pose(pose_array: np.ndarray) -> np.ndarray:
    """Devuelve la pose espejada horizontalmente.

    - Invierte la coordenada x: x_new = 1 - x_old.
    - Swappea los indices de izquierda/derecha para que las etiquetas
      anatomicas sigan siendo coherentes despues del espejado visual.
    """
    mirrored = pose_array.copy()
    mirrored[:, 0] = 1.0 - mirrored[:, 0]
    for a, b in LEFT_RIGHT_PAIRS:
        mirrored[[a, b]] = mirrored[[b, a]]
    return mirrored


@dataclass
class FrameScore:
    """Resultado de comparar un frame de jugador contra uno de referencia."""
    score: float                 # 0 a 100
    angle_errors: np.ndarray     # error absoluto por articulacion (grados)
    valid: bool                  # True si la comparacion fue confiable


def landmarks_to_array(landmarks) -> np.ndarray:
    """Convierte landmarks de MediaPipe a array (33, 4): x, y, z, visibility."""
    arr = np.zeros((NUM_LANDMARKS, 4), dtype=np.float32)
    for i, lm in enumerate(landmarks):
        arr[i] = [lm.x, lm.y, lm.z, lm.visibility]
    return arr


def compute_angle(p_a: np.ndarray, p_v: np.ndarray, p_b: np.ndarray) -> float:
    """Calcula el angulo en grados en `p_v` formado por los segmentos a->v y v->b.

    Usa solo coordenadas 2D (x, y) para robustez frente a ruido en z.
    """
    v1 = p_a[:2] - p_v[:2]
    v2 = p_b[:2] - p_v[:2]
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return 0.0
    cos_theta = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_theta)))


def extract_angles(pose_array: np.ndarray) -> np.ndarray:
    """Dado un array (33, 4) de landmarks, devuelve un vector (NUM_ANGLES,)."""
    angles = np.zeros(NUM_ANGLES, dtype=np.float32)
    for i, (_, a, v, b) in enumerate(JOINT_ANGLES):
        angles[i] = compute_angle(pose_array[a], pose_array[v], pose_array[b])
    return angles


def compare_poses(
    player_pose: np.ndarray,
    reference_pose: np.ndarray,
    visibility_threshold: float = 0.4,
    tolerance_degrees: float = 30.0,
) -> FrameScore:
    """Compara la pose del jugador contra la de referencia.

    Args:
        player_pose: array (33, 4) con landmarks del jugador.
        reference_pose: array (33, 4) con landmarks de referencia.
        visibility_threshold: minima visibilidad para considerar valida una articulacion.
        tolerance_degrees: error en grados que se mapea al 0% del score por articulacion.
            Por debajo de este umbral el score crece linealmente hasta 100%.

    Returns:
        FrameScore con score promedio (0-100), errores por articulacion y validez.
    """
    player_angles = extract_angles(player_pose)
    reference_angles = extract_angles(reference_pose)

    angle_errors = np.abs(player_angles - reference_angles)

    # Solo cuentan articulaciones donde ambos lados son visibles.
    valid_mask = np.zeros(NUM_ANGLES, dtype=bool)
    for i, (_, a, v, b) in enumerate(JOINT_ANGLES):
        player_ok = (
            player_pose[a, 3] >= visibility_threshold
            and player_pose[v, 3] >= visibility_threshold
            and player_pose[b, 3] >= visibility_threshold
        )
        ref_ok = (
            reference_pose[a, 3] >= visibility_threshold
            and reference_pose[v, 3] >= visibility_threshold
            and reference_pose[b, 3] >= visibility_threshold
        )
        valid_mask[i] = player_ok and ref_ok

    if not valid_mask.any():
        return FrameScore(score=0.0, angle_errors=angle_errors, valid=False)

    # Score por articulacion: 100% si el error es 0, 0% si supera la tolerancia.
    per_joint_score = np.clip(1.0 - (angle_errors / tolerance_degrees), 0.0, 1.0)
    avg_score = float(per_joint_score[valid_mask].mean()) * 100.0

    return FrameScore(score=avg_score, angle_errors=angle_errors, valid=True)


def score_to_label(score: float) -> str:
    """Mapea un score numerico a una etiqueta tipo Just Dance."""
    if score >= 90:
        return "PERFECT"
    if score >= 75:
        return "GREAT"
    if score >= 55:
        return "GOOD"
    if score >= 30:
        return "OK"
    return "MISS"

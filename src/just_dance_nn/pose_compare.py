"""
Comparador de poses basado en angulos articulares.

La idea: dos personas con distinta altura, complexion o distancia a la camara
producen coordenadas (x, y) muy distintas aun haciendo el mismo movimiento.
En cambio, los ANGULOS entre articulaciones (codo, hombro, cadera, rodilla)
son invariantes a esas diferencias y capturan la pose real.

Formato CANONICO de pose (13 keypoints) -- comun a todos los detectores
(MediaPipe, PoseNet de la Jetson, etc.). Cada backend mapea su salida nativa
a este formato. Una pose es un array (13, 3): [x, y, visibility], con x/y
normalizados a [0, 1] y visibility en [0, 1].

    Indice  Nombre
    0       nose
    1       left_shoulder    2   right_shoulder
    3       left_elbow       4   right_elbow
    5       left_wrist       6   right_wrist
    7       left_hip         8   right_hip
    9       left_knee       10   right_knee
    11      left_ankle      12   right_ankle
"""

import numpy as np

# ---------------------------------------------------------------------------
# Formato canonico de keypoints (compartido por todos los backends)
# ---------------------------------------------------------------------------

CANONICAL_NAMES = [
    "nose",            # 0
    "left_shoulder",   # 1
    "right_shoulder",  # 2
    "left_elbow",      # 3
    "right_elbow",     # 4
    "left_wrist",      # 5
    "right_wrist",     # 6
    "left_hip",        # 7
    "right_hip",       # 8
    "left_knee",       # 9
    "right_knee",      # 10
    "left_ankle",      # 11
    "right_ankle",     # 12
]

NUM_KEYPOINTS = len(CANONICAL_NAMES)
VIS_IDX = 2  # columna de visibility en el array (x=0, y=1, visibility=2)

# Indices por nombre, para legibilidad
NOSE = 0
L_SHOULDER, R_SHOULDER = 1, 2
L_ELBOW, R_ELBOW = 3, 4
L_WRIST, R_WRIST = 5, 6
L_HIP, R_HIP = 7, 8
L_KNEE, R_KNEE = 9, 10
L_ANKLE, R_ANKLE = 11, 12

# Articulaciones a evaluar: (nombre, punto_a, vertice, punto_b)
# El angulo es el formado en `vertice` por los segmentos vertice->a y vertice->b
JOINT_ANGLES = [
    ("codo_izq",    L_SHOULDER, L_ELBOW, L_WRIST),   # hombro - codo - muneca (izq)
    ("codo_der",    R_SHOULDER, R_ELBOW, R_WRIST),   # hombro - codo - muneca (der)
    ("hombro_izq",  L_ELBOW, L_SHOULDER, L_HIP),     # codo - hombro - cadera (izq)
    ("hombro_der",  R_ELBOW, R_SHOULDER, R_HIP),     # codo - hombro - cadera (der)
    ("cadera_izq",  L_SHOULDER, L_HIP, L_KNEE),      # hombro - cadera - rodilla (izq)
    ("cadera_der",  R_SHOULDER, R_HIP, R_KNEE),      # hombro - cadera - rodilla (der)
    ("rodilla_izq", L_HIP, L_KNEE, L_ANKLE),         # cadera - rodilla - tobillo (izq)
    ("rodilla_der", R_HIP, R_KNEE, R_ANKLE),         # cadera - rodilla - tobillo (der)
]

NUM_ANGLES = len(JOINT_ANGLES)

# Pares izquierda/derecha para swappear al espejar (la nariz se mantiene).
LEFT_RIGHT_PAIRS = [
    (L_SHOULDER, R_SHOULDER),
    (L_ELBOW, R_ELBOW),
    (L_WRIST, R_WRIST),
    (L_HIP, R_HIP),
    (L_KNEE, R_KNEE),
    (L_ANKLE, R_ANKLE),
]

# Conexiones para dibujar el esqueleto (en indices canonicos).
POSE_CONNECTIONS = [
    (L_SHOULDER, R_SHOULDER),
    (L_SHOULDER, L_ELBOW), (L_ELBOW, L_WRIST),
    (R_SHOULDER, R_ELBOW), (R_ELBOW, R_WRIST),
    (L_SHOULDER, L_HIP), (R_SHOULDER, R_HIP),
    (L_HIP, R_HIP),
    (L_HIP, L_KNEE), (L_KNEE, L_ANKLE),
    (R_HIP, R_KNEE), (R_KNEE, R_ANKLE),
]


class FrameScore:
    """Resultado de comparar un frame de jugador contra uno de referencia.

    (Clase simple en vez de @dataclass para ser compatible con Python 3.6,
    que es el que corre en la Jetson; dataclasses recien existe desde 3.7.)
    """

    def __init__(self, score, angle_errors, valid):
        self.score = score              # 0 a 100
        self.angle_errors = angle_errors  # error absoluto por articulacion (grados)
        self.valid = valid              # True si la comparacion fue confiable


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


def compute_angle(p_a: np.ndarray, p_v: np.ndarray, p_b: np.ndarray) -> float:
    """Calcula el angulo en grados en `p_v` formado por los segmentos a->v y v->b.

    Usa solo coordenadas 2D (x, y).
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
    """Dado un array (13, 3) de keypoints, devuelve un vector (NUM_ANGLES,)."""
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
        player_pose: array (13, 3) con keypoints del jugador.
        reference_pose: array (13, 3) con keypoints de referencia.
        visibility_threshold: minima visibilidad para considerar valida una articulacion.
        tolerance_degrees: error en grados que se mapea al 0% del score por articulacion.
            Por debajo de este umbral el score crece linealmente hasta 100%.

    Returns:
        FrameScore con score promedio (0-100), errores por articulacion y validez.
    """
    player_angles = extract_angles(player_pose)
    reference_angles = extract_angles(reference_pose)

    angle_errors = np.abs(player_angles - reference_angles)

    # Solo cuentan articulaciones donde los 3 puntos son visibles en ambas poses.
    valid_mask = np.zeros(NUM_ANGLES, dtype=bool)
    for i, (_, a, v, b) in enumerate(JOINT_ANGLES):
        player_ok = (
            player_pose[a, VIS_IDX] >= visibility_threshold
            and player_pose[v, VIS_IDX] >= visibility_threshold
            and player_pose[b, VIS_IDX] >= visibility_threshold
        )
        ref_ok = (
            reference_pose[a, VIS_IDX] >= visibility_threshold
            and reference_pose[v, VIS_IDX] >= visibility_threshold
            and reference_pose[b, VIS_IDX] >= visibility_threshold
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

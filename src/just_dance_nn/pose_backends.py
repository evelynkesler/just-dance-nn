"""
Detectores de pose intercambiables ("backends").

Todos los backends implementan la misma interfaz `PoseDetector` y devuelven
poses en el FORMATO CANONICO de 13 keypoints definido en `pose_compare.py`
(array (13, 3) = [x, y, visibility], x/y normalizados a [0, 1]).

Asi el resto del proyecto (comparador, motor de juego, extractor) NO sabe ni
le importa que red neuronal hay debajo:

    - "mediapipe": MediaPipe Pose (BlazePose, Google). Corre en CPU.
      Sirve para desarrollar y probar en cualquier PC/Mac.
    - "posenet":   PoseNet de NVIDIA (jetson-inference). Corre en la GPU de la
      Jetson Nano con TensorRT. Es el backend de produccion del TP.

La eleccion se hace con un flag --backend en los scripts. El codigo del juego
es identico en los dos; solo cambia el detector.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from just_dance_nn.pose_compare import CANONICAL_NAMES, NUM_KEYPOINTS

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"


class PoseDetector:
    """Interfaz comun a todos los detectores de pose."""

    def detect(self, frame_bgr: np.ndarray, timestamp_ms: int = 0) -> list[np.ndarray]:
        """Detecta personas en un frame BGR (formato de OpenCV).

        Devuelve una lista de poses, cada una un array (13, 3) en formato
        canonico. Lista vacia si no se detecto a nadie.
        """
        raise NotImplementedError

    def close(self) -> None:
        """Libera recursos. Opcional."""


# ---------------------------------------------------------------------------
# Backend MediaPipe (CPU) -- para desarrollar/probar en la Mac o cualquier PC
# ---------------------------------------------------------------------------

# canonical[i] = landmark de MediaPipe en este indice
# (MediaPipe Pose tiene 33 landmarks; tomamos los 13 que nos interesan)
_MP_TO_CANONICAL = [
    0,    # nose
    11,   # left_shoulder
    12,   # right_shoulder
    13,   # left_elbow
    14,   # right_elbow
    15,   # left_wrist
    16,   # right_wrist
    23,   # left_hip
    24,   # right_hip
    25,   # left_knee
    26,   # right_knee
    27,   # left_ankle
    28,   # right_ankle
]


class MediaPipePoseDetector(PoseDetector):
    """Detector basado en MediaPipe Pose Landmarker (API tasks)."""

    def __init__(self, num_poses: int = 1, model_complexity: str = "lite",
                 running_mode: str = "video") -> None:
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision

        self._mp = mp
        model_path = MODELS_DIR / f"pose_landmarker_{model_complexity}.task"
        if not model_path.exists():
            raise SystemExit(
                f"No existe el modelo MediaPipe: {model_path}\n"
                f"Descargalo con: uv run python scripts/download_model.py "
                f"--complexity {model_complexity}"
            )

        mode = {
            "video": mp_vision.RunningMode.VIDEO,
            "image": mp_vision.RunningMode.IMAGE,
        }[running_mode]

        base_options = mp_python.BaseOptions(model_asset_path=str(model_path))
        options = mp_vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=mode,
            num_poses=num_poses,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._landmarker = mp_vision.PoseLandmarker.create_from_options(options)
        self._running_mode = running_mode

    def detect(self, frame_bgr: np.ndarray, timestamp_ms: int = 0) -> list[np.ndarray]:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        if self._running_mode == "video":
            result = self._landmarker.detect_for_video(mp_image, timestamp_ms)
        else:
            result = self._landmarker.detect(mp_image)
        if not result.pose_landmarks:
            return []
        return [self._to_canonical(lms) for lms in result.pose_landmarks]

    @staticmethod
    def _to_canonical(landmarks) -> np.ndarray:
        arr = np.zeros((NUM_KEYPOINTS, 3), dtype=np.float32)
        for i, mp_idx in enumerate(_MP_TO_CANONICAL):
            lm = landmarks[mp_idx]
            arr[i] = [lm.x, lm.y, lm.visibility]
        return arr

    def close(self) -> None:
        self._landmarker.close()


# ---------------------------------------------------------------------------
# Backend PoseNet (GPU) -- para la Jetson Nano con TensorRT
# ---------------------------------------------------------------------------

class PoseNetPoseDetector(PoseDetector):
    """Detector basado en PoseNet de NVIDIA (jetson-inference).

    Solo funciona en una Jetson (necesita GPU/TensorRT). El import de
    jetson_inference es perezoso para que este modulo se pueda importar en
    la Mac sin romper.
    """

    def __init__(self, num_poses: int = 1, model: str = "resnet18-body",
                 threshold: float = 0.15) -> None:
        try:
            from jetson_inference import poseNet
            from jetson_utils import cudaFromNumpy
        except ImportError as exc:  # pragma: no cover (solo en Jetson)
            raise SystemExit(
                "El backend 'posenet' requiere jetson-inference, que solo corre "
                "en la Jetson. Instalalo segun la guia (Etapa 2). En la Mac usa "
                "--backend mediapipe."
            ) from exc

        self._cudaFromNumpy = cudaFromNumpy
        self._net = poseNet(model, threshold=threshold)
        self._num_poses = num_poses

    def detect(self, frame_bgr: np.ndarray, timestamp_ms: int = 0) -> list[np.ndarray]:
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        cuda_img = self._cudaFromNumpy(rgb)
        poses = self._net.Process(cuda_img, overlay="none")

        results: list[np.ndarray] = []
        for pose in poses:
            arr = np.zeros((NUM_KEYPOINTS, 3), dtype=np.float32)
            for i, name in enumerate(CANONICAL_NAMES):
                idx = pose.FindKeypoint(name)
                if idx >= 0:
                    kp = pose.Keypoints[idx]
                    # PoseNet da coords en pixeles -> normalizamos a [0, 1].
                    # visibility = 1.0 si el keypoint fue detectado, 0.0 si no.
                    arr[i] = [kp.x / w, kp.y / h, 1.0]
            results.append(arr)
        return results


# ---------------------------------------------------------------------------
# Fabrica
# ---------------------------------------------------------------------------

def create_detector(backend: str, num_poses: int = 1,
                    model_complexity: str = "lite",
                    running_mode: str = "video") -> PoseDetector:
    """Crea el detector segun el backend pedido.

    backend: "mediapipe" (Mac/PC, CPU) o "posenet" (Jetson, GPU).
    """
    backend = backend.lower()
    if backend == "mediapipe":
        return MediaPipePoseDetector(
            num_poses=num_poses,
            model_complexity=model_complexity,
            running_mode=running_mode,
        )
    if backend == "posenet":
        return PoseNetPoseDetector(num_poses=num_poses)
    raise SystemExit(f"Backend desconocido: {backend!r}. Usa 'mediapipe' o 'posenet'.")

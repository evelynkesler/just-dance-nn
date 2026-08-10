#!/usr/bin/env bash
# =============================================================================
# jetson_run.sh  --  Correr Just Dance NN en la Jetson (backend posenet, GPU)
#
# Se corre DENTRO del contenedor jetson-inference (o en una Jetson con
# jetson-inference instalado nativo), parado en la carpeta del proyecto.
#
# En la Jetson NO se usa uv ni mediapipe: el backend 'posenet' usa
# jetson_inference (ya instalado) + cv2 + numpy. Los scripts se importan como
# paquete, asi que fijamos PYTHONPATH=src.
#
# Uso:
#   bash jetson_run.sh menu                          # MENU: elegir cancion de videos/ y jugar
#   bash jetson_run.sh poc                           # PoC: esqueleto en GPU (valida posenet)
#   bash jetson_run.sh play  VIDEO.mp4 REF.npz [START END]   # juego con un video puntual
#   bash jetson_run.sh verify VIDEO.mp4 REF.npz [START]      # verificar referencia
#
# Ejemplos:
#   bash jetson_run.sh menu
#   bash jetson_run.sh poc
#   bash jetson_run.sh play "videos/Lush Life.mp4" "references/Lush Life.npz"
# =============================================================================
set -e

# Parado en la raiz del proyecto (donde estan src/, models/, este script).
cd "$(dirname "$0")"
export PYTHONPATH="src:${PYTHONPATH}"

PY="${PYTHON:-python3}"
BACKEND="posenet"   # en la Jetson siempre GPU. (Para depurar en CPU: PYTHON=... BACKEND=mediapipe)
SUB="${1:-poc}"; shift || true

# Chequeo rapido de que estamos en una Jetson con la red disponible.
if ! "$PY" -c "import jetson_inference" 2>/dev/null; then
  echo "!! No se pudo importar jetson_inference."
  echo "   Estas dentro del contenedor jetson-inference? (./docker/run.sh ...)"
  echo "   o lo tenes instalado nativo en la Jetson?"
  exit 1
fi

case "$SUB" in
  menu)
    echo ">> Menu de canciones (escanea videos/). Elegi con W/S o flechas, ENTER para jugar."
    "$PY" src/just_dance_nn/play.py --backend "$BACKEND" --fullscreen --infer-every 2
    ;;

  poc)
    echo ">> PoC: esqueleto en vivo con PoseNet (GPU). Q para salir."
    "$PY" src/just_dance_nn/pose_poc.py --backend "$BACKEND"
    ;;

  play)
    VIDEO="$1"; REF="$2"; START="$3"; END="$4"
    if [ -z "$VIDEO" ] || [ -z "$REF" ]; then
      echo "Uso: bash jetson_run.sh play VIDEO.mp4 REF.npz [START END]"; exit 1
    fi
    ARGS=(--reference "$REF" --backend "$BACKEND")
    [ -n "$START" ] && ARGS+=(--start "$START")
    [ -n "$END" ]   && ARGS+=(--end "$END")
    echo ">> Juego completo en GPU. Espacio=pausa, Q=salir."
    "$PY" src/just_dance_nn/play.py "$VIDEO" "${ARGS[@]}"
    ;;

  verify)
    VIDEO="$1"; REF="$2"; START="$3"
    if [ -z "$VIDEO" ] || [ -z "$REF" ]; then
      echo "Uso: bash jetson_run.sh verify VIDEO.mp4 REF.npz [START]"; exit 1
    fi
    ARGS=(--reference "$REF")
    [ -n "$START" ] && ARGS+=(--start "$START")
    echo ">> Reproduciendo video con el esqueleto de la referencia superpuesto."
    "$PY" src/just_dance_nn/verify_reference.py "$VIDEO" "${ARGS[@]}"
    ;;

  *)
    echo "Subcomando desconocido: $SUB"
    echo "Usa: menu | poc | play | verify"
    exit 1
    ;;
esac

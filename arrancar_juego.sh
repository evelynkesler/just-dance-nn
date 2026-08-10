#!/usr/bin/env bash
# =============================================================================
# arrancar_juego.sh  --  Arranque de UN SOLO PASO para Just Dance NN
#
# Para usar cuando la Jetson YA ESTA configurada (jetson-inference instalado y
# el proyecto en ~/Desktop/just-dance-nn). Hace todo solo:
#   1. Da permiso de pantalla al contenedor.
#   2. Entra al contenedor con el proyecto montado.
#   3. Abre directamente el MENU de canciones.
#
# Se corre en la TERMINAL NORMAL de la Jetson (NO dentro del contenedor):
#   bash ~/Desktop/just-dance-nn/arrancar_juego.sh
#
# Cuando salis del menu (Q), el contenedor se cierra solo.
# =============================================================================
set -e

PROJECT_DIR="$HOME/Desktop/just-dance-nn"   # donde esta el proyecto
JI_DIR="$HOME/jetson-inference"             # donde esta jetson-inference

# --- Chequeos amigables ---
if [ ! -d "$JI_DIR" ]; then
  echo "ERROR: no encuentro jetson-inference en $JI_DIR"
  echo "       Hay que instalarlo una vez (ver PASOS_JETSON.md)."
  exit 1
fi
if [ ! -d "$PROJECT_DIR" ]; then
  echo "ERROR: no encuentro el proyecto en $PROJECT_DIR"
  echo "       Copialo del pendrive a esa ruta."
  exit 1
fi

# --- 1) Permiso de pantalla para el contenedor (corre como root) ---
xhost +local:root >/dev/null 2>&1 || true

# --- 2) y 3) Entrar al contenedor y abrir el menu directamente ---
echo ">> Arrancando Just Dance NN (menu de canciones)..."
echo "   (la 1ra cancion sin referencia se extrae sola; Q para salir)"
cd "$JI_DIR"
./docker/run.sh --volume "$PROJECT_DIR:/just-dance-nn" \
  bash -c "cd /just-dance-nn && bash jetson_run.sh menu"

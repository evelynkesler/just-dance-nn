#!/usr/bin/env bash
# =============================================================================
# setup_jetson.sh  --  Preparacion de la Jetson Nano (Etapa 2)
# Se corre UNA sola vez, en la terminal de la Jetson (NO dentro del contenedor).
#
# Instala jetson-inference (PoseNet / resnet18-body, GPU + TensorRT) via Docker,
# que es la forma mas simple en JetPack 4.6. Despues entras al contenedor con
# tu proyecto montado y usas jetson_run.sh.
# =============================================================================
set -e

PROJECT_DIR="${1:-$HOME/just-dance-nn}"   # donde copiaste el proyecto del pendrive
JI_DIR="$HOME/jetson-inference"

echo ">> 1/3  Dependencias base..."
sudo apt-get update
sudo apt-get install -y git

echo ">> 2/3  Clonando jetson-inference (si no existe)..."
if [ ! -d "$JI_DIR" ]; then
  git clone --recursive --depth=1 https://github.com/dusty-nv/jetson-inference "$JI_DIR"
else
  echo "   Ya existe $JI_DIR, lo salteo."
fi

echo ">> 3/3  Bajando el contenedor (la 1ra vez tarda varios minutos)..."
cd "$JI_DIR"
# docker/run.sh baja la imagen, ofrece descargar modelos (marcar resnet18-body)
# y te deja una shell adentro. Montamos el proyecto en /just-dance-nn.
echo ""
echo "============================================================"
echo " Listo. Ahora entra al contenedor con tu proyecto montado:"
echo ""
echo "   cd $JI_DIR"
echo "   ./docker/run.sh --volume $PROJECT_DIR:/just-dance-nn"
echo ""
echo " Y YA DENTRO del contenedor:"
echo "   cd /just-dance-nn"
echo "   bash jetson_run.sh poc          # validar PoseNet en la GPU"
echo "============================================================"

# Guía del proyecto — Just Dance NN en Jetson Nano

> Documento para repasar con el grupo y el profe en clase.
> Proyecto: clon educativo de *Just Dance* con detección de pose por red neuronal.
> Placa: **NVIDIA Jetson Nano Developer Kit (P3450)** — la clásica de 2019, 4 GB.
> Última actualización: 2026-06-07.

---

## 1. ¿Qué hace el proyecto?

1. **Extrae la pose** de un bailarín de un video de referencia (offline).
2. **Captura la pose del jugador** por webcam en tiempo real.
3. **Compara las dos poses** frame a frame usando **ángulos articulares**
   (codos, hombros, caderas, rodillas) y devuelve un **puntaje**.

La comparación se hace por ángulos (no por coordenadas) porque el ángulo de un
codo a 90° es el mismo sin importar la altura, complexión o distancia a la
cámara de la persona. Es invariante → mide *la pose real*, no el tamaño.

---

## 2. La decisión técnica importante: CAMINO B

La "red neuronal" del proyecto es el detector de pose. Hay dos opciones:

| | Camino A: MediaPipe | **Camino B: PoseNet (NVIDIA)** ✅ ELEGIDO |
|---|---|---|
| Red neuronal | BlazePose (Google) | ResNet18-body (NVIDIA) |
| ¿Dónde corre? | CPU | **GPU de la Jetson, con TensorRT** |
| ¿Aprovecha la Jetson? | No | **Sí — ese es el sentido de tener una Jetson** |
| Instalación en la Nano | Difícil (Python 3.6 vs 3.11) | **Nativa en JetPack 4.6** |
| Puntos del cuerpo | 33 | 18 |

**Por qué B:** el objetivo de la materia es correr la red en la Jetson. Si usáramos
MediaPipe en CPU, cualquier computadora serviría y la Jetson no aportaría nada.
PoseNet corre en la **GPU con TensorRT**, que es justo lo que hace especial a la
placa. Además se instala nativo en la Jetson, evitando el lío de versiones de Python.

**Lo bueno:** el diseño del juego NO cambia. Solo cambia el detector de pose por
debajo. La lógica de comparación por ángulos se mantiene; solo hay que reasignar
qué punto es cada articulación (de 33 puntos de MediaPipe a 18 de PoseNet).

### ❓ Preguntas para confirmar con el profe

1. ¿Confirma que quiere la inferencia en la **GPU con TensorRT** (PoseNet de
   NVIDIA), y no MediaPipe en CPU?
2. ¿Está OK usar el repo **`jetson-inference`** de NVIDIA como base de la red, o
   pide entrenar/usar otra red puntual?
3. ¿Espera que midamos y mostremos el rendimiento (FPS en GPU)?

---

## 3. Hardware — checklist

| Item | ¿Lo tenemos? |
|---|---|
| Jetson Nano P3450 | ✅ |
| Fuente de alimentación 5V | ✅ |
| **microSD 256 GB** | ✅ |
| Webcam USB | ✅ |
| Cable HDMI + monitor | ✅ (confirmar monitor) |
| Mouse + teclado USB | ✅ |
| **Jumper para los pines J48** (solo si la fuente es barrel jack) | ⚠️ verificar |

> **Sobre la fuente:** si es de conector redondo (barrel jack) de 5V/4A, la Jetson
> no la usa hasta poner un jumper en los pines **J48** de la placa. Si la fuente es
> microUSB, no hace falta jumper.

---

## 4. Plan en 4 etapas

```
ETAPA 1            ETAPA 2              ETAPA 3            ETAPA 4
Encender    →    PoseNet en la    →   Portar el     →   Juego completo
la Jetson        GPU (demo)           código            + optimización
```

- **Etapa 1 y 2** dependen solo de la Jetson.
- **Etapa 3** (código) se puede escribir en la Mac, pero se **prueba en la Jetson**
  (PoseNet solo corre ahí, necesita la GPU).

---

## 5. ETAPA 1 — Encender la Jetson

**Se hace desde la Mac. Lo único físico que falta es flashear la microSD.**

1. **Descargar la imagen** de JetPack 4.6.1 (~6,5 GB). [Ya descargándose en `~/Downloads/`]
   Link oficial: https://developer.nvidia.com/jetson-nano-sd-card-image
2. **Instalar Balena Etcher** en la Mac (gratis): https://etcher.balena.io
3. **Flashear:** abrir Etcher → *Flash from file* → elegir el `.zip` →
   *Select target* → la microSD → *Flash!*. Tarda ~15-20 min. **No descomprimir el zip**, Etcher lo lee directo.
4. **Montar la Jetson:** meter la microSD en la ranura, conectar monitor (HDMI),
   teclado, mouse, webcam, y por último la **fuente** (poner el jumper J48 si es barrel jack).
5. **Primer arranque:** se inicia un **Ubuntu** normal. Aceptar licencia, crear
   usuario y contraseña, conectar a wifi. La partición se expande sola a los 256 GB.
6. ✅ Ya tenemos una mini computadora Linux con GPU andando.

> Si la pantalla queda negra: revisar que el monitor sea HDMI directo (no por
> adaptador raro) y que la fuente sea suficiente (la webcam consume).

---

## 6. ETAPA 2 — PoseNet corriendo en la GPU

**Todo esto se hace en la Jetson, en su terminal.**

### 6.1 Instalar jetson-inference (vía Docker, lo más simple)

```bash
sudo apt-get update
sudo apt-get install -y git
git clone --recursive --depth=1 https://github.com/dusty-nv/jetson-inference
cd jetson-inference
docker/run.sh        # baja el contenedor y entra; la 1ra vez tarda
```

> Alternativa sin Docker (compilar): `mkdir build && cd build && cmake ../ &&
> make -j$(nproc) && sudo make install && sudo ldconfig`. En el `cmake` aparece un
> menú para bajar modelos: **marcar los de pose (resnet18-body)**.

### 6.2 Correr el demo de pose con la webcam

```bash
# /dev/video0 es la webcam USB. Si no anda, probar /dev/video1.
posenet.py /dev/video0
```

Debería abrirse una ventana con el **esqueleto dibujado sobre vos en vivo,
corriendo en la GPU**. 🎉 Este es el primer hito demostrable del Camino B.

> PoseNet da 18 puntos del cuerpo con nombres: `nose`, `left_shoulder`,
> `right_shoulder`, `left_elbow`, `left_wrist`, `left_hip`, `left_knee`,
> `left_ankle`, etc. Se acceden por nombre con `pose.FindKeypoint('left_elbow')`.

---

## 7. ETAPA 3 — Detector intercambiable ✅ YA HECHO

**Esta etapa ya está implementada y probada en la Mac.** El código se
refactorizó para que el detector de pose sea un "módulo enchufable": la lógica
del juego es idéntica y solo cambia la red por debajo.

### Cómo quedó la arquitectura

```
  DETECTOR (cambia)                 LÓGICA DEL JUEGO (siempre igual)
  ┌──────────────────┐             ┌────────────────────────────┐
  │ backend mediapipe│──┐          │ comparar ángulos, score,   │
  │  (CPU, Mac/PC)   │  ├─► pose ─►│ HUD, combo, esqueleto      │
  │ backend posenet  │──┘  (13,3)  │                            │
  │  (GPU, Jetson)   │             └────────────────────────────┘
  └──────────────────┘
```

- **`pose_backends.py`** (nuevo): define la interfaz `PoseDetector` y los dos
  backends. `create_detector("mediapipe" | "posenet")` elige cuál usar.
- **Formato canónico de 13 keypoints** (en `pose_compare.py`): cada backend
  traduce su salida nativa (33 de MediaPipe / 18 de PoseNet) a este formato común.
- Todos los scripts aceptan ahora **`--backend mediapipe|posenet`**.

### Cómo se corre

```bash
# En la Mac (desarrollo / demo):
uv run python src/just_dance_nn/pose_poc.py                      # PoC esqueleto
uv run python src/just_dance_nn/extract_reference.py VIDEO ...   # extraer referencia
uv run python src/just_dance_nn/play.py VIDEO --reference REF ...# jugar

# En la Jetson (mismo código, GPU):
python src/just_dance_nn/pose_poc.py --backend posenet
python src/just_dance_nn/play.py VIDEO --reference REF --backend posenet
```

### Las 8 articulaciones (idénticas en los dos backends)

| Ángulo | Punto A | Vértice | Punto B |
|---|---|---|---|
| Codo izq/der | shoulder | elbow | wrist |
| Hombro izq/der | elbow | shoulder | hip |
| Cadera izq/der | shoulder | hip | knee |
| Rodilla izq/der | hip | knee | ankle |

Las 8 existen tanto en los 33 puntos de MediaPipe como en los 18 de PoseNet.

### Lo único que falta validar en la Jetson
El backend `posenet` está escrito pero **solo se puede probar en la Jetson**
(necesita GPU/TensorRT). Al correrlo allá puede requerir ajustes menores: los
nombres exactos de los keypoints de PoseNet y cómo `Process()` devuelve las
coordenadas. Es el único punto a verificar en la placa.

---

## 8. ETAPA 4 — Juego completo + optimización

- Correr el Just Dance entero en la Jetson.
- Medir FPS y, si hace falta, bajar resolución o usar el modelo más liviano.
- (Roadmap original) audio sincronizado, selector de canciones, pantalla completa.

---

## 9. Estado actual (lo que ya está hecho en la Mac)

- ✅ Repo clonado y entorno instalado (`uv sync`).
- ✅ Demo de MediaPipe funcionando en la Mac (sirve para mostrar el *concepto*).
- ✅ Decisión Camino B tomada y justificada.
- ✅ **Arquitectura de detector intercambiable implementada y probada en la Mac**
  (Etapa 3 hecha): backend mediapipe + backend posenet, formato canónico de 13
  keypoints, flag `--backend`. Tests del comparador y del backend MediaPipe OK.
- ✅ Imagen de JetPack 4.6.1 descargada (`~/Downloads/jetson-nano-jp461-sd-card-image.zip`).
- ⬜ Flashear microSD → arrancar Jetson (Etapa 1).
- ⬜ Instalar jetson-inference + correr PoseNet (Etapa 2).
- ⬜ Validar el backend posenet en la Jetson y correr el juego (Etapa 4).

---

## 10. Resumen para la presentación

> "Hicimos un clon de Just Dance que detecta la pose del jugador con una red
> neuronal y la compara con un bailarín de referencia por ángulos articulares.
> Decidimos correr la inferencia en la **GPU de la Jetson Nano con TensorRT**
> usando **PoseNet de NVIDIA**, para aprovechar el hardware de la placa en vez de
> usar la CPU. Tenemos el concepto andando y la Jetson en proceso de puesta a punto."

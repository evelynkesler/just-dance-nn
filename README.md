# Just Dance NN

Clon educativo de Just Dance basado en redes neuronales para detección de pose. Reproduce un video de baile de referencia, captura los movimientos del jugador por webcam y devuelve un puntaje de precisión basado en la similitud entre la pose del jugador y la del bailarín del video.

Trabajo Práctico Integrador para la materia **Redes Neuronales** (UTN FRD), pensado para ejecutarse en una **NVIDIA Jetson Nano** (portable, no requiere PC).

---

## Tabla de contenidos

- [Descripción técnica](#descripción-técnica)
- [Arquitectura](#arquitectura)
- [Requisitos](#requisitos)
- [Instalación](#instalación)
- [Cómo usarlo](#cómo-usarlo)
- [Estructura del proyecto](#estructura-del-proyecto)
- [Detalles del comparador](#detalles-del-comparador)
- [Roadmap / TODO](#roadmap--todo)

---

## Descripción técnica

El sistema combina tres etapas:

1. **Extracción offline** de los keypoints de pose del video de referencia, con tracking de un único bailarín (el principal) entre todas las personas que aparezcan en pantalla.
2. **Captura en tiempo real** de la pose del jugador por webcam.
3. **Comparación frame a frame** entre la pose del jugador y la del bailarín de referencia, usando ángulos articulares como métrica invariante a altura, distancia y complexión.

La red neuronal utilizada es **MediaPipe Pose Landmarker** (modelo basado en BlazePose), que devuelve 33 keypoints 3D del cuerpo por persona. Se usa la API moderna `mediapipe.tasks` (no la API legacy `solutions`).

---

## Arquitectura

```
┌────────────────────────┐
│  extract_reference.py  │  ← procesa video offline, hace tracking
│  (modo offline)        │     del bailarín principal y guarda .npz
└──────────┬─────────────┘
           │
           ▼
   references/<song>.npz  (landmarks pre-extraídos)
           │
           ▼
┌────────────────────────┐         ┌──────────────┐
│       play.py          │ ◄───────┤   webcam     │
│  (motor de juego)      │         └──────────────┘
│                        │
│  ┌──────────────────┐  │
│  │ pose_compare.py  │  │  ← compara ángulos articulares
│  │ (comparador)     │  │
│  └──────────────────┘  │
└──────────┬─────────────┘
           │
           ▼
    UI: video referencia | webcam con esqueleto | HUD score/combo
           │
           ▼
    Pantalla final con estrellas y gráfico de precisión
```

### Módulos

| Archivo | Función |
|---|---|
| `pose_poc.py` | PoC inicial: solo detecta y dibuja la pose en la webcam. Útil para verificar que el modelo funciona. |
| `extract_reference.py` | Procesa un video y guarda los landmarks del bailarín principal en un `.npz`. Permite seleccionar al bailarín con un click. |
| `verify_reference.py` | Reproduce el video con el esqueleto extraído superpuesto, para verificar visualmente que se trackeó al bailarín correcto. |
| `pose_compare.py` | Lógica de comparación de poses por ángulos articulares + utilidades (mirror, label de score). |
| `play.py` | Motor de juego completo: video sincronizado, captura webcam, comparación en vivo, HUD, score final. |

---

## Requisitos

- **Python 3.11** o superior
- **[uv](https://docs.astral.sh/uv/)** (gestor de entornos y dependencias)
- **Webcam** funcional
- **~6 GB libres** (el venv pesa unos 250 MB; el resto es para videos y modelos)
- **No usar OneDrive ni Dropbox** como ubicación del proyecto — los launchers `.exe` del venv se rompen con la sincronización en la nube. Usar una carpeta local (ej: `C:\Users\<usuario>\Documents\`).

### Sistema operativo

- Probado en Windows 11 con PowerShell.
- Linux/Mac debería funcionar igual (el `cv2.CAP_DSHOW` es exclusivo de Windows; en Linux se usa V4L2 automáticamente cambiando esa línea o eliminándola).

---

## Instalación

### 1. Instalar uv

```powershell
# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Clonar el repo

```bash
git clone <URL_DEL_REPO> just-dance-nn
cd just-dance-nn
```

### 3. Instalar dependencias

```bash
uv sync
```

`uv` se encarga de:
- Descargar Python 3.11 si no lo tenés.
- Crear el `.venv/` en la raíz del proyecto.
- Instalar todas las dependencias del `uv.lock` (versiones exactas reproducibles).

### 4. Descargar el modelo de MediaPipe

```bash
uv run python scripts/download_model.py
```

Esto descarga `pose_landmarker_lite.task` (~5.6 MB) en `models/`.

### 5. Conseguir un video de baile

**No incluimos videos en el repo por copyright.** Cada uno baja el suyo desde YouTube:

```bash
uv run yt-dlp -f "best[height<=720]" -o "Video 1 - Happy.mp4" "<URL_YOUTUBE>"
```

Recomendaciones para elegir video:
- Plano fijo (sin zoom ni pan).
- El bailarín principal bien visible.
- Resolución 720p (más alta es lento de procesar).
- Duración 1-3 minutos.

---

## Cómo usarlo

### Paso 1: extraer la referencia

```bash
uv run python src/just_dance_nn/extract_reference.py "Video 1 - Happy.mp4" --start 36 --end 207
```

- `--start` y `--end` (en segundos) recortan al rango donde está el baile (saltean intro/outro).
- Si el video tiene varios bailarines, una ventana te pedirá **click sobre el principal**. Después se trackea esa persona en todos los frames por proximidad espacial.
- Salida: `references/Video 1 - Happy.npz` con los landmarks por frame.

### Paso 2 (opcional): verificar la extracción

```bash
uv run python src/just_dance_nn/verify_reference.py "Video 1 - Happy.mp4" --reference "references/Video 1 - Happy.npz" --start 36
```

Reproduce el video con el esqueleto superpuesto. Si el esqueleto sigue siempre al bailarín correcto → todo bien. Si salta a otros bailarines → revisar tracking.

### Paso 3: jugar

```bash
uv run python src/just_dance_nn/play.py "Video 1 - Happy.mp4" --reference "references/Video 1 - Happy.npz" --start 36 --end 207
```

Controles:
- **Espacio** → pausa / reanudar (también pausa el audio)
- **F** → pantalla completa on/off
- **Q** → salir antes de tiempo

Flags útiles:
- `--no-mirror` → desactiva el modo espejo (por defecto, el video y los landmarks de referencia se espejan para que el jugador imite "lado por lado" como en Just Dance real).
- `--fullscreen` → arranca en pantalla completa.
- `--no-audio` → sin sonido. (El audio usa `ffplay`, que viene con **ffmpeg**: `brew install ffmpeg` en Mac. Si no está instalado, el juego avisa y sigue mudo.)
- `--countdown N` → segundos de cuenta regresiva (default 3).
- `--cam-index N` → si tenés varias cámaras.

---

## Estructura del proyecto

```
just-dance-nn/
├── src/just_dance_nn/
│   ├── pose_poc.py           # PoC de detección de pose
│   ├── pose_compare.py       # Algoritmo de comparación
│   ├── extract_reference.py  # Extractor offline de referencias
│   ├── verify_reference.py   # Verificador visual
│   └── play.py               # Motor de juego principal
├── scripts/
│   └── download_model.py     # Descarga el modelo de MediaPipe
├── models/                   # (gitignored) modelos .task
├── references/               # (gitignored) landmarks .npz
├── pyproject.toml
├── uv.lock                   # versiones exactas, NO editar a mano
└── README.md
```

---

## Detalles del comparador

### Por qué ángulos articulares y no coordenadas

Comparar coordenadas brutas (x, y) entre referencia y jugador funciona mal: penaliza diferencias de altura, complexión y distancia a la cámara, aunque el movimiento sea idéntico.

Los **ángulos articulares** son invariantes a esas diferencias. Si la bailarina del video tiene su codo a 90° y el jugador también, hacen el mismo gesto sin importar el resto.

### Articulaciones evaluadas

8 ángulos en total:

| Articulación | Vértice | Segmentos |
|---|---|---|
| Codo izq/der | codo | hombro → codo → muñeca |
| Hombro izq/der | hombro | codo → hombro → cadera |
| Cadera izq/der | cadera | hombro → cadera → rodilla |
| Rodilla izq/der | rodilla | cadera → rodilla → tobillo |

### Mapeo a score

Por cada ángulo:
- error de **0°** → 100 % en esa articulación
- error ≥ **30°** → 0 % en esa articulación
- valor lineal entremedias

Score del frame = promedio de las articulaciones visibles (las que tienen los 3 puntos involucrados con `visibility ≥ 0.4`).

### Etiquetas

| Score promedio | Etiqueta |
|---|---|
| ≥ 90 | PERFECT |
| ≥ 75 | GREAT  |
| ≥ 55 | GOOD   |
| ≥ 30 | OK     |
| < 30 | MISS   |

### Modo espejo

En el juego real de Just Dance, el video del bailarín se muestra **espejado** para que el jugador imite "lado por lado" en pantalla. Implementamos esto con `mirror_pose()` en `pose_compare.py`, que invierte la coordenada x y swappea los índices izquierdo/derecho de los landmarks anatómicos para mantener la correspondencia.

---

## Roadmap / TODO

### Funcional

- [x] PoC de pose detection en webcam
- [x] Extracción de referencia con tracking de un solo bailarín
- [x] Comparador por ángulos articulares con modo espejo
- [x] Motor de juego con HUD, combo, estrellas y pantalla final
- [x] **Audio sincronizado** (`ffplay` como subproceso, pausa con SIGSTOP/SIGCONT, fallback mudo si no hay ffmpeg)
- [ ] **Esqueleto guía** dibujado sobre el bailarín de la referencia
- [x] **Selector de canciones** (menú que escanea `videos/`; videos nuevos aparecen solos)
- [ ] **Tolerancia temporal**: comparar contra ±N frames y quedarse con el mejor match (más perdonador)
- [ ] **Multi-jugador** con identificación persistente (tracking de identidad entre frames)

### Despliegue

- [x] Optimizar inferencia para Jetson Nano (backend `posenet`: jetson-inference + TensorRT en GPU)
- [x] Modo pantalla completa (`--fullscreen` / tecla F)
- [x] Empaquetar como script de arranque para la Jetson (`arrancar_juego.sh`)

### Académico

- [ ] Informe final
- [ ] Video de presentación

---

## Créditos

- Modelo de pose: [MediaPipe Pose Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker) (Google)
- Framework de detección: [MediaPipe Tasks API](https://ai.google.dev/edge/mediapipe)

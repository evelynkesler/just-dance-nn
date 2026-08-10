# Paso a paso en la Jetson Nano — Just Dance NN

> Guía operativa para correr el proyecto en la **Jetson Nano** con el backend
> `posenet` (PoseNet de NVIDIA, GPU + TensorRT). Seguir los pasos en orden.
> Cada `[ ]` es algo para tildar cuando lo termines.

---

## 0. Antes de ir a la Jetson (esto se hace en la Mac)

El juego completo necesita un **video** y su **referencia `.npz`** (los keypoints
del bailarín, pre-extraídos). Eso se genera en la Mac con MediaPipe; la Jetson no
hace falta para este paso. La validación de PoseNet (paso 5) NO necesita esto, así
que podés avanzar igual y dejar el juego completo para cuando tengas el video.

```bash
# en la Mac, dentro de just-dance-nn/
uv run yt-dlp -f "best[height<=720]" -o "Video.mp4" "<URL_YOUTUBE>"
uv run python src/just_dance_nn/extract_reference.py "Video.mp4" --start 36 --end 207
# -> genera references/Video.npz   (el .npz es agnóstico del backend: sirve para posenet)
```

### Qué copiar al pendrive

Copiá la carpeta `just-dance-nn/` **sin** lo pesado/inútil para la placa:

| Copiar | Por qué |
|---|---|
| ✅ `src/just_dance_nn/` (los `.py`) | El código del juego |
| ✅ `setup_jetson.sh`, `jetson_run.sh`, `arrancar_juego.sh` | Los scripts de arranque |
| ✅ `README.md`, `GUIA_JETSON.md`, `PASOS_JETSON.md` | Referencia para la defensa |
| ✅ `videos_jetson/*.mp4` → copiarlos como `videos/` | Versiones 480p livianas de decodificar (¡NO llevar los originales 720p/1080p: en la placa se ven lentos!) |
| ✅ `references/*.npz` | Las referencias pre-extraídas (sirven igual para los videos 480p: mismos fps y frames) |
| ❌ `models/pose_landmarker_lite.task` | Es de MediaPipe (CPU), no se usa con posenet |
| ❌ `uv.lock`, `pyproject.toml`, `.venv/` | `uv` no se usa en la Jetson |
| ❌ los PDF | No hacen falta en la placa |

Comando para armar la copia limpia (ajustá `NOMBRE_PENDRIVE`):

```bash
rsync -av --exclude='.venv' --exclude='models' --exclude='__pycache__' \
  --exclude='uv.lock' --exclude='*.pdf' \
  "just-dance-nn" "/Volumes/NOMBRE_PENDRIVE/"
```

- [ ] Video y referencia extraídos en la Mac
- [ ] Proyecto copiado al pendrive

---

## 0.5 Fluidez: checklist de rendimiento en la Jetson

Para que el juego se vea rápido y fluido en la placa:

1. **Modo máxima potencia** (fuera del contenedor, una vez por arranque; requiere la fuente de 4A):
   ```bash
   sudo nvpmodel -m 0      # modo MAXN
   sudo jetson_clocks      # relojes al máximo
   ```
2. **Usar los videos 480p** de `videos_jetson/` (copiados como `videos/` en la placa). Decodificar 720p/1080p por CPU era la causa principal del video lento.
3. **`--infer-every 2`**: la red corre 1 de cada 2 frames y reusa la pose entre medio. Ya viene activado en `jetson_run.sh menu`.
4. Cerrar Chromium y cualquier otra app pesada antes de jugar.

---

## 1. Encender la Jetson

- [ ] microSD con JetPack 4.6 puesta
- [ ] Monitor (HDMI), teclado, mouse y **webcam USB** conectados
- [ ] Fuente conectada (si es barrel jack, **jumper en J48**)
- [ ] Arranca Ubuntu y llegás al escritorio

---

## 2. Pasar el proyecto del pendrive a la Jetson

```bash
cp -r /media/$USER/NOMBRE_PENDRIVE/just-dance-nn ~/just-dance-nn
cd ~/just-dance-nn
```

- [ ] Carpeta `~/just-dance-nn` con los `.py` y los `.sh` adentro

---

## 3. Instalar jetson-inference (Etapa 2) — una sola vez

```bash
cd ~/just-dance-nn
bash setup_jetson.sh
```

Esto instala git, clona `jetson-inference` y baja el contenedor Docker. Cuando
ofrezca descargar modelos, **marcá `resnet18-body`** (es el de PoseNet). La
primera vez tarda varios minutos.

- [ ] `jetson-inference` clonado en `~/jetson-inference`
- [ ] Modelo `resnet18-body` descargado

---

## 4. Entrar al contenedor con el proyecto montado

```bash
cd ~/jetson-inference
./docker/run.sh --volume ~/just-dance-nn:/just-dance-nn
# ya DENTRO del contenedor:
cd /just-dance-nn
```

> Todos los pasos siguientes (5 en adelante) se corren **dentro del contenedor**.
> Si la webcam no es `/dev/video0`, probá `/dev/video1`.

- [ ] Estás dentro del contenedor, parada en `/just-dance-nn`

---

## 5. Validar PoseNet en la GPU (hito clave del TP)

```bash
bash jetson_run.sh poc
```

Debería abrirse una ventana con **tu esqueleto dibujado en vivo, corriendo en la
GPU**. Salir con `Q`.

> Si falla por nombres de keypoints o por cómo `Process()` devuelve las
> coordenadas, es el único ajuste fino esperable del backend posenet
> (ver `pose_backends.py`, clase `PoseNetPoseDetector`). Los nombres canónicos
> (`left_shoulder`, `left_elbow`, …) ya coinciden con los de PoseNet.

- [ ] Veo mi esqueleto en vivo → **backend posenet validado, Etapa 2 lista** ✅

---

## 6. Jugar — menú de canciones

La forma recomendada es el **menú**: escanea la carpeta `videos/` y te deja
elegir. Cualquier video nuevo que pongas en `videos/` aparece solo como opción.

```bash
bash jetson_run.sh menu
```

- Movés la selección con **W/S** o las **flechas**, elegís con **ENTER**.
- Cada canción muestra **LISTO** (ya tiene referencia) o **FALTA REF.**
- Si elegís una sin referencia, **se extrae sola** la primera vez (tarda un poco)
  y después arranca. Las próximas veces ya entra directo.
- Al terminar una canción, volvés al menú.

Controles dentro del juego: **Espacio** = pausa · **Q** = salir (vuelve al menú).

> Alternativa con un video puntual (sin menú):
> `bash jetson_run.sh play "videos/Lush Life.mp4" "references/Lush Life.npz"`

- [ ] El menú lista mis videos y puedo jugar uno en la Jetson

---

## 7. Medir FPS para el informe (la evidencia de que usás la GPU)

Al jugar (paso 6), el HUD muestra abajo a la derecha:

```
MAX COMBO 12   |   28 FPS   |   red 45 FPS
```

- `28 FPS` = loop completo (webcam + red + dibujo)
- `red 45 FPS` = **solo la red neuronal** → esto mide la GPU

Al **cerrar** el juego, la consola imprime el resumen para copiar al informe:

```
--- Rendimiento de inferencia (posenet) ---
  frames procesados : 5123
  latencia media    : 22.1 ms  ->  45.2 FPS
  latencia p95      : 28.4 ms
  latencia maxima   : 61.0 ms
```

### Comparación GPU vs CPU (el argumento del "Camino B")

El número que más vende en la defensa es **`red ... FPS` en posenet (GPU)** y
contrastarlo con MediaPipe en CPU. Como en la Jetson normalmente no se puede
instalar MediaPipe (Python 3.6), la comparación práctica es:

| Plataforma | Backend | Dónde se corre |
|---|---|---|
| Jetson Nano | `posenet` (GPU/TensorRT) | en la placa, paso 6 |
| Mac/PC | `mediapipe` (CPU) | en la Mac: `uv run python src/just_dance_nn/play.py ...` |

Anotá la **latencia media / FPS de la red** de cada uno. Dejar claro en el
informe que son plataformas distintas (no es una comparación 100% limpia, pero
muestra el orden de magnitud que aporta la GPU).

- [ ] FPS de inferencia anotados (posenet en Jetson)
- [ ] FPS de inferencia anotados (mediapipe en Mac) para comparar

---

## 8. Cierre

- [ ] Screenshots / video del juego corriendo en la Jetson (para la presentación)
- [ ] Números de FPS en el informe
- [ ] Guion de presentación (ver punto 10 de `GUIA_JETSON.md`)

---

## Apéndice — Problemas comunes

| Síntoma | Qué probar |
|---|---|
| No hay sonido | El audio usa `ffplay` (parte de ffmpeg). Dentro del contenedor: `apt-get update && apt-get install -y ffmpeg` (se pierde al salir del contenedor; repetir por sesión o aceptar jugar sin audio). Si no está, el juego sigue mudo sin romperse. |
|---|---|
| Pantalla negra al arrancar | HDMI directo (sin adaptadores raros); fuente suficiente |
| `No se pudo abrir la webcam` | Probar `/dev/video1`; revisar que la webcam esté enchufada antes de entrar al contenedor |
| `No se pudo importar jetson_inference` | No estás dentro del contenedor; volver al paso 4 |
| FPS muy bajos | Bajar resolución de webcam; usar el video más liviano; cerrar otras apps |
| El esqueleto salta o falta puntos | Ajuste fino de `PoseNetPoseDetector` en `pose_backends.py` (nombres/coords) |

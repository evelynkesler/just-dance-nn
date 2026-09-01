# Trabajo Práctico Integrador — Redes Neuronales

## Just Dance NN: juego de baile con detección de pose en tiempo real sobre NVIDIA Jetson Nano

**UTN FRD — Redes Neuronales — Año 2026 — Grupo 2**

**Integrantes:** Basaldua Bernardo · Costa Lautaro · Juarez Juan · Kesler Evelyn · Natuche Lucas · Nuñez Imanol · Redondo Alejo · Rey Mateo

**Repositorio:** *(link al repo)*

---

## 1. Resumen

Just Dance NN es un clon educativo de Just Dance: el sistema reproduce un video
de baile de referencia, captura al jugador por webcam, estima su pose corporal
con una red neuronal en tiempo real y devuelve un puntaje de similitud entre
ambas poses, con la estética y mecánica del juego original (etiquetas
PERFECT/GREAT/GOOD/OK/MISS, combos, estrellas y desglose final de precisión).

El proyecto corre completo en una **NVIDIA Jetson Nano** (4 GB), usando la GPU
de la placa mediante **PoseNet optimizada con TensorRT**, y también en una
PC/Mac convencional usando **MediaPipe Pose Landmarker (BlazePose)** sobre CPU.
Esta dualidad no fue un capricho: fue la solución a una limitación real de la
plataforma embebida, y permitió además comparar el rendimiento de inferencia
CPU vs GPU con el mismo juego como banco de pruebas.

---

## 2. Objetivo

Dado cualquier video de baile (por ejemplo, descargado de YouTube), permitir
que un jugador se pare frente a una cámara y obtenga en tiempo real un puntaje
de cuán parecido baila respecto del bailarín del video, corriendo sobre
hardware embebido (Jetson Nano) de forma portable: sin PC, sin internet, con un
solo comando de arranque.

---

## 3. Redes neuronales utilizadas

### 3.1 MediaPipe Pose Landmarker (BlazePose) — PC/Mac, CPU

- Modelo de Google basado en **BlazePose**; devuelve **33 keypoints 3D** del
  cuerpo con un valor de confianza (visibility) por punto.
- Se usa la API moderna `mediapipe.tasks` en modo video.
- Rol en el proyecto: (a) motor de juego en PC, (b) **extracción offline de
  referencias** (procesar el video del bailarín y guardar sus poses).

### 3.2 PoseNet (resnet18-body) — Jetson Nano, GPU + TensorRT

- Red de estimación de postura de **jetson-inference** (NVIDIA), con backbone
  ResNet-18, **pre-optimizada con TensorRT** para la GPU Maxwell de la Nano.
- Devuelve 18 keypoints (formato COCO/OpenPose) con confianza por punto.
- Rol en el proyecto: motor de juego en la Jetson, corriendo en GPU.

### 3.3 Por qué dos redes: la limitación que definió la arquitectura

MediaPipe **no puede ejecutarse en la Jetson Nano**: la imagen oficial
(JetPack 4.6) trae Ubuntu 18.04 con Python 3.6, y MediaPipe requiere
Python 3.8+ y no publica binarios para ARM64/Jetson. Se evaluó compilar
MediaPipe desde fuente con soporte GPU (delegate de TF Lite), pero implicaba
semanas de trabajo con alto riesgo de incompatibilidad (la Nano usa CUDA 10.2 y
MediaPipe moderno apunta a CUDA 11+).

**Solución adoptada:** una capa de abstracción (`pose_backends.py`) con dos
detectores intercambiables por flag (`--backend mediapipe|posenet`) que
convergen a un **formato canónico común de 13 keypoints** (nariz, hombros,
codos, muñecas, caderas, rodillas y tobillos, con confianza por punto). Todo el
resto del sistema (comparador, juego, referencias) es agnóstico del backend.
Consecuencia clave: **las referencias extraídas en PC con MediaPipe funcionan
sin cambios en la Jetson con PoseNet.**

---

## 4. Arquitectura del sistema

```
[offline, en PC]                            [tiempo real, en la Jetson]

video de baile (.mp4)                        webcam del jugador
        |                                            |
extract_reference.py                          PoseNet (GPU+TensorRT)
  MediaPipe multi-persona                            |
  + tracking del bailarín principal           pose canónica (13 kp)
        |                                            |
referencia .npz  ---------------------->  play.py (motor de juego)
  (poses por frame)                          comparador de ángulos
                                             con tolerancia temporal
                                                     |
                                       UI: video | webcam+esqueleto | HUD
                                       score, combos, pantalla final
```

**Módulos principales** (`src/just_dance_nn/`):

| Módulo | Función |
|---|---|
| `pose_backends.py` | Abstracción de las dos redes; formato canónico de keypoints |
| `extract_reference.py` | Procesa un video offline, trackea al bailarín principal y guarda sus poses en `.npz` |
| `verify_reference.py` | Reproduce el video con el esqueleto extraído superpuesto (control visual) |
| `pose_compare.py` | Comparación de poses por ángulos articulares; modo espejo; ventana temporal vectorizada |
| `play.py` | Motor de juego: menú de canciones, sincronización, audio, HUD, scoring, pantalla final |
| `pose_poc.py` | Prueba de concepto: esqueleto en vivo (sirve para validar cada backend) |

---

## 5. Problemas encontrados y soluciones

### 5.1 Score erróneo por diferencias físicas

**Problema.** Comparando coordenadas (x, y) brutas, dos personas haciendo el
mismo movimiento obtenían puntajes muy distintos por altura, complexión o
distancia a la cámara.

**Solución.** Comparación por **ángulos articulares**, invariantes a escala y
posición: se evalúan 8 ángulos (codos, hombros, caderas y rodillas, lados
izquierdo y derecho), cada uno definido por 3 keypoints. Solo cuentan las
articulaciones cuyos 3 puntos superan un umbral de confianza (visibility ≥ 0.4)
en ambas poses — es decir, se usan los metadatos de confianza de la red, no
solo sus coordenadas.

### 5.2 MediaPipe no corre en la Jetson

Descripto en §3.3: backend intercambiable con formato canónico común.

### 5.3 Caída de framerate en la Jetson

**Problema.** El juego se veía lento y entrecortado en la placa.

**Diagnóstico.** Se instrumentó la latencia de inferencia por separado del loop
completo, y el hallazgo fue contraintuitivo: **el cuello de botella principal
no era la red neuronal (que corre en GPU) sino la decodificación del video de
referencia**, que OpenCV hace por CPU (sin aceleración de hardware) — con
videos 1080p, los 4 núcleos ARM A57 de la Nano no dan abasto.

**Soluciones aplicadas (acumulativas):**
1. **Videos re-codificados a 480p** (H.264 baseline) para la placa, con la
   misma cantidad de frames y fps → las referencias `.npz` siguen siendo
   válidas. Fue la mejora de mayor impacto.
2. **Preprocesamiento offline de la referencia**: las poses del bailarín se
   extraen una sola vez en PC; en vivo solo se infiere sobre la webcam
   (elimina la mitad de la carga de inferencia).
3. **Inferencia por frames clave** (`--infer-every N`): la red corre 1 de cada
   N frames y entre medio se reutiliza la última pose (la pose humana casi no
   cambia en 60 ms). Con N=2 el loop casi duplica su fluidez.
4. **Avance por `grab()`** en vez de seeks aleatorios para mantener el video
   sincronizado con el reloj del juego (un seek reabre desde el keyframe y era
   una fuente de tirones).
5. Modo de máxima potencia de la placa (`nvpmodel -m 0` + `jetson_clocks`).

### 5.4 El puntaje castigaba el retraso humano, no el error del paso

**Problema.** Comparando contra el frame exacto de la referencia, un jugador
que hacía el paso correcto ~300 ms tarde (el retraso natural de reacción)
recibía MISS. En pruebas reales el score quedaba clavado en ~25/100 aunque el
jugador acertara la coreografía.

**Solución.** **Tolerancia temporal**: se compara la pose del jugador contra
una ventana de frames de la referencia (0.5 s hacia atrás y ~0.17 s hacia
adelante) y se toma el mejor match. Para que fuera viable en la Jetson se
vectorizó con numpy: los ángulos de toda la referencia se pre-calculan al
cargar la canción, y la evaluación de la ventana completa cuesta **~56 µs por
frame** (la primera implementación, con un loop en Python, costaba 6.4 ms).

### 5.5 Calibración del score

Dos ajustes surgidos de pruebas con jugadores reales:

- **Score honesto:** los frames donde el jugador no es detectado ahora valen 0
  en el promedio final (antes se ignoraban: no bailar podía dar puntaje > 0).
- **Zona de gracia:** el jitter de la red más las diferencias de cámara
  producen 10–15° de error de base aunque el paso esté clavado. Con la curva
  lineal original (0% a los 30°) el máximo humano alcanzable rondaba 60/100.
  La curva final: error ≤ 12° = 100% por articulación, decayendo linealmente
  hasta 0% a los 45°.

**Resultado medido:** el mismo jugador, con la misma calidad de baile, pasó de
25 a ~70 puntos; y el puntaje responde a la práctica (10 intentos consecutivos
de la misma canción: 53 → 70.5, con el desglose mostrando la mejora por
articulación).

### 5.6 Una estatua sacaba PERFECT

**Problema.** La comparación es de poses estáticas: un jugador parado quieto
en pose neutra (brazos colgando) puntuaba alto cada vez que la coreografía
pasaba cerca de una pose parecida, y la ventana temporal —que elige el mejor
frame del rango— lo amplificaba. Simulado contra las 6 referencias: media de
77–86/100 y hasta 39 % de frames PERFECT sin mover un dedo.

**Solución.** **Gate de movimiento** (`movement_factor()` en
`pose_compare.py`): se compara cuánto se movió cada articulación (rango del
ángulo en los últimos 0.5 s) del jugador contra lo que se movió la referencia.
Si la coreografía se mueve (> 20°) y el jugador no, el score del frame se
multiplica por un factor que baja hasta 0.15. Al movimiento del jugador se le
descuentan 10° de jitter de detección, y alcanza con el 40 % de la amplitud de
la referencia para no ser penalizado (no hace falta clavar la amplitud del
bailarín profesional). Complemento: la ventana temporal se achicó de 0.5 s a
0.3 s, porque era el principal amplificador del problema.

**Resultado medido (simulación):** la "estatua" pasó de media 77–86 a 9–17
(94–99 % de MISS); un "bailarín bueno" simulado (la propia referencia con
250 ms de retraso y jitter de cámara) conserva media 91–96 con 70–87 % de
PERFECT.

### 5.7 Mover una mano con medio cuerpo fuera de cámara daba PERFECT

**Problema.** El score promedia solo las articulaciones visibles en ambas
poses. Con las piernas fuera del encuadre, el jugador quedaba evaluado
únicamente por brazos y hombros: saludando con una mano se veían frames de
PERFECT (en simulación, el score instantáneo tocaba 100 en varias canciones).

**Solución.** **Regla de cobertura**: si el jugador no muestra el cuerpo
completo (todas las articulaciones que la referencia ofrece en ese frame), el
frame vale 0 (MISS) — no se puede validar un paso que no se ve. Dos detalles
de diseño:

- **Gracia de 0.4 s:** la detección parpadea (una rodilla desaparece 1–2
  frames aunque el jugador esté entero en cámara); la regla solo se aplica si
  la cobertura incompleta se sostiene. Sin esto, cada parpadeo cortaba el
  combo con un MISS injusto.
- **El denominador es lo que ofrece la referencia**, no las 8 articulaciones:
  si el video de la coreografía tiene un plano cerrado (solo torso), comparar
  solo el torso es todo lo que se puede hacer y no se castiga al jugador.

Además el juego muestra el aviso "CUERPO INCOMPLETO EN CAMARA" sobre el panel
del jugador para que corrija el encuadre.

**Resultado medido (simulación):** "saludando sin piernas" pasó de media
16–28 con frames PERFECT a media 0–4 con 97–100 % de MISS (los pocos frames
no-MISS coinciden con planos cerrados de la referencia, donde la comparación
de torso sí es válida); el bailarín de cuerpo entero no cambia, y con
parpadeos artificiales de visibilidad cada 2 s pierde < 1 % de frames.

---

## 6. Características del juego final

- **Menú de canciones** que escanea la carpeta `videos/`: cualquier video nuevo
  aparece automáticamente; si no tiene referencia, se extrae sola la primera vez.
- **Audio sincronizado** por subproceso, con cadena de fallback
  `ffplay → mpv → GStreamer` (GStreamer viene en el contenedor de la Jetson,
  lo que permite música sin internet ni instalaciones).
- **Modo espejo** (como el Just Dance real): el video y sus poses se espejan
  para que el jugador imite "lado por lado" lo que ve.
- HUD en vivo: score instantáneo, promedio, combo, etiquetas animadas, barra de
  progreso, FPS del loop y FPS de la red.
- **Pantalla final con desglose**: estrellas, % de frames por etiqueta,
  **precisión por grupo articular** (codos/hombros/caderas/rodillas), % del
  tiempo detectado, gráfico de evolución y tecla R para reintentar.
- Pantalla completa (`--fullscreen` / tecla F), pausa (espacio), salida (Q).
- **Arranque de un solo comando en la Jetson** (`arrancar_juego.sh`): entra al
  contenedor, monta el proyecto y abre el menú. Funciona 100% offline.
- Biblioteca actual: 6 canciones con referencia pre-extraída.

---

## 7. Resultados y mediciones

### 7.1 Calidad de la extracción de referencias (tracking del bailarín)

| Video | Características | Frames trackeados |
|---|---|---|
| Macarena | plano fijo, 1 bailarín | **100%** |
| Shuffle | plano fijo, 1 bailarín | **100%** |
| Single Ladies | plano fijo, 1 bailarín | **100%** |
| On The Floor | videoclip | 91% |
| Beauty And A Beat | videoclip | 89% |
| Lush Life | videoclip con cortes de cámara | 69% |

Conclusión: los planos fijos con el bailarín visible trackean perfecto; los
cortes de cámara y zooms son la principal fuente de pérdida.

### 7.2 Rendimiento de inferencia (la red sola)

| Plataforma | Red / backend | Latencia media | FPS de inferencia |
|---|---|---|---|
| MacBook (M2, CPU) | MediaPipe Pose Landmarker lite | 11–16 ms | **~63–91 FPS** |
| Jetson Nano (GPU) | PoseNet resnet18-body + TensorRT | *(completar)* | *(completar)* |

*(Los números de la Jetson se completan con la medición en la placa: el HUD
muestra los FPS de la red en vivo y al cerrar cada canción la consola imprime
latencia media, p95 y máxima.)*

Nota metodológica: son plataformas distintas (no es una comparación
controlada); el objetivo es mostrar el orden de magnitud y que la Nano logra
tiempo real gracias a la GPU con un modelo que su CPU no podría correr a esa
tasa.

### 7.3 Experiencia de juego medida (PC)

- Detección del jugador: 90–100% del tiempo de juego.
- Progresión de score con la práctica: 53 → 70.5 en 10 intentos.
- El desglose por articulación diagnostica errores reales: p. ej., jugar
  demasiado cerca de la cámara deja fuera caderas/rodillas y el sistema lo
  reporta (`caderas --`), explicando el score bajo.

---

## 8. Cómo se usa

### En PC/Mac

```bash
uv sync                                    # instala dependencias
uv run python scripts/download_model.py    # baja el modelo de MediaPipe
uv run python src/just_dance_nn/play.py    # abre el menú y a jugar
```

### En la Jetson Nano (offline)

```bash
# instalación (una sola vez): clona jetson-inference y baja el modelo
bash setup_jetson.sh          # marcar resnet18-body

# jugar (cada vez): un solo comando
bash arrancar_juego.sh
```

Flags útiles de `play.py`: `--backend mediapipe|posenet`, `--fullscreen`,
`--no-audio`, `--no-mirror`, `--infer-every N`, `--tolerance S`,
`--cam-index N`, `--countdown N`.

Guías completas en el repo: `README.md`, `PASOS_JETSON.md`, `GUIA_JETSON.md`.

---

## 9. Conclusiones

- **Es factible un juego de pose en tiempo real en una Jetson Nano**, pero no
  alcanza con "tener GPU": hizo falta elegir una red optimizada para la placa
  (PoseNet/TensorRT), sacar toda la carga posible del camino crítico
  (referencias offline, videos livianos, inferencia por frames clave) y medir
  para encontrar el cuello de botella real (que resultó ser la decodificación
  de video, no la red).
- **La abstracción de backends fue la decisión de diseño más rentable**:
  permitió desarrollar y probar en PC con MediaPipe y desplegar en la Jetson
  con PoseNet sin tocar el resto del sistema, y convirtió una limitación de la
  plataforma en una comparación CPU vs GPU medible.
- **La métrica importa tanto como la red**: la salida cruda del modelo no
  produce un buen juego. Los ángulos articulares (invariancia física), el uso
  de la confianza por keypoint, la tolerancia temporal (retraso humano) y la
  calibración de la curva de score fueron lo que transformó "detección de
  pose" en "juego jugable y justo".

## 10. Trabajo futuro

- Esqueleto guía dibujado sobre el bailarín de la referencia.
- Multi-jugador con identificación persistente de cada jugador.
- Decodificación de video por hardware en la Jetson (GStreamer/NVDEC) para
  usar videos de mayor resolución.
- Modo "clases": repetir un fragmento en loop hasta dominarlo.

---

*Anexo sugerido: capturas del juego (menú, partida, pantalla final), foto del
setup con la Jetson, y los resúmenes de consola con las métricas de rendimiento
de cada plataforma.*

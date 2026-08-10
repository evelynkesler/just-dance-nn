# Esta aplicacion usa una red pre-entrenada especializada para la deteccion de botellas, entrenada 
#     con una versión modificada del dataset MS COCO filtrado para enfocarse específicamente 
#     en la detección de botellas y latas de bebidas. 
# Esto se hace para que el modelo ocupe menos espacio.
#  
import jetson_inference
import jetson_utils

# Cargamos específicamente la red de botellas
net = jetson_inference.detectNet("coco-bottle", threshold=0.5)


# camera = jetson_utils.videoSource("csi://0")
# 'csi://0' es para la cámara oficial de la Jetson (Raspberry Pi V2)
# Si usas una cámara Web USB, cambia esto por '/dev/video0'
camera = jetson_utils.videoSource("/dev/video0")

display = jetson_utils.videoOutput("display://0")

while display.IsStreaming():
    img = camera.Capture()
    if img is None: continue

    # Detectará únicamente botellas y dibujará el cuadro sobre ellas
    detections = net.Detect(img)
    
    display.Render(img)
    display.SetStatus(f"Bottle Detection | FPS: {net.GetNetworkFPS():.1f}")
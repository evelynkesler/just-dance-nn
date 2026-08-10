# Esta aplicacion usa una red ssd-mobilenet-v2 (creada por investigadores de Google Research)
#   pre-entrenada paa la deteccion de objetos.
# 
import jetson_inference
import jetson_utils
import sys

def main():
    # 1. Cargar la red de detección de objetos
    # Usamos 'ssd-mobilenet-v2' por defecto (es rápida y ligera para la Nano)
    # 'threshold' es la confianza mínima (0.5 = 50%) para detectar un objeto
    net = jetson_inference.detectNet("ssd-mobilenet-v2", threshold=0.5)

    # 2. Inicializar la cámara
    # 'csi://0' es para la cámara oficial de la Jetson (Raspberry Pi V2)
    # Si usas una cámara Web USB, cambia esto por '/dev/video0'
    camera = jetson_utils.videoSource("/dev/video0")

    # 3. Inicializar la ventana para mostrar el resultado
    display = jetson_utils.videoOutput("display://0")

    print("Procesando stream de video... Presiona Ctrl+C para salir.")

    # 4. Bucle principal de procesamiento
    while display.IsStreaming():
        # Capturar el siguiente frame de la cámara
        img = camera.Capture()
        
        if img is None:
            continue

        # Pasar la imagen a la red neuronal
        # Esto detecta los objetos y dibuja automáticamente los cuadros (overlays) en la imagen
        detections = net.Detect(img)

        # (Opcional) Imprimir en la terminal lo que se está detectando
        print(f"Objetos detectados en este frame: {len(detections)}")
        for detection in detections:
            print(f" - Clase ID: {detection.ClassID}, Confianza: {detection.Confidence:.2f}")

        # Renderizar la imagen modificada en la pantalla
        display.Render(img)

        # Actualizar el título de la ventana con los FPS (cuadros por segundo)
        display.SetStatus(f"Object Detection | Network FPS: {net.GetNetworkFPS():.1f}")

if __name__ == "__main__":
    main()
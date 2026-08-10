# Esta aplicacion usa una red pre-entrenada resnet18 (creada por un equipo de investigadores de Microsoft Research)
#  especializada en la eteccion de poses del cuerpo humano.
# 
import jetson_inference
import jetson_utils

def main():
    # 1. Cargar la red de estimación de pose
    # Usamos 'resnet18-body', pero puedes cambiarlo por 'mobilenet-v1-body' si buscas más velocidad
    # 'threshold' es la confianza mínima (0.15 es buena para no perder articulaciones al moverse)
    net = jetson_inference.poseNet("resnet18-body", threshold=0.15)

    # 2. Inicializar la cámara (CSI por defecto, cambiar a "/dev/video0" si usas una webcam USB)
    # 'csi://0' es para la cámara oficial de la Jetson (Raspberry Pi V2)
    camera = jetson_utils.videoSource("/dev/video0")

    # 3. Inicializar la ventana de visualización
    display = jetson_utils.videoOutput("display://0")

    print("Procesando estimación de pose... Presiona Ctrl+C para salir.")

    while display.IsStreaming():
        # Capturar el frame de la cámara
        img = camera.Capture()
        
        if img is None:
            continue

        # Procesar la imagen para encontrar las poses
        # El método 'Perform' detecta las personas y dibuja automáticamente 
        # las líneas del esqueleto y los círculos de las articulaciones sobre 'img'
        poses = net.Process(img)

        # (Opcional) Analizar los datos numéricos en la terminal
        print(f"Se detectaron {len(poses)} personas en este frame.")
        for pose in poses:
            # Puedes acceder a cada punto clave individualmente si lo necesitas
            # por ejemplo, para detectar si alguien levantó un brazo
            print(f" - Pose ID: {pose.ID} | Cantidad de puntos clave detectados: {len(pose.Keypoints)}")
            
            # Ejemplo: Buscar dónde está la nariz de la persona
            for keypoint in pose.Keypoints:
                if keypoint.ID == net.FindKeypointID("nose"):
                    print(f"   -> Posición de la Nariz: X={keypoint.x:.1f}, Y={keypoint.y:.1f}")

        # Renderizar la imagen modificada en la pantalla
        display.Render(img)

        # Mostrar los FPS en la barra de título de la ventana
        display.SetStatus(f"Pose Estimation | Network FPS: {net.GetNetworkFPS():.1f}")

if __name__ == "__main__":
    main()
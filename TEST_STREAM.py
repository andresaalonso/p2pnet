import cv2
import time
from P2PNetDetector.P2PNet import P2PNet




def analyze_video_stream(video_source=0, frame_interval=2, all_frames=False):



    detector = P2PNet(backbone="vgg16_bn")
    cap = cv2.VideoCapture(video_source)
    
    if not cap.isOpened():
        print("Error: No se pudo abrir el video.")
        return
    
    last_processed_time = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Fin del video o error en la captura.")
            break
        
        current_time = time.time()
        if all_frames or current_time - last_processed_time >= frame_interval:
            last_processed_time = current_time
            
            # Calcular el tiempo que tarda en realizar la detección
            inicio = time.time_ns()
            frame, points = detector.run(frame)
            final = time.time_ns() - inicio
            print(f"Detection: {final/1000000}ms")
            
            detector.draw_predictions(frame, points)
        
            # Mostrar el video en tiempo real
            cv2.imshow("Video Stream", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Como el análisis de cada frame con la red tarda un poco
    # Le he puesto que tome solo un frame cada cierto tiempo y asi no se pete
    # frame_interval es la cantidad de segundos que deja entre cada vez que pilla un frame
    # Podeis bajarlo hasta que veais que empieza a petar
    analyze_video_stream(video_source="/home/andres/Escritorio/P2PNET_ROOT/was/Whats_001.mp4", frame_interval=1.2, all_frames=True)

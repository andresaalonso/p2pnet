import cv2
import time
import json
from P2PNetDetector import P2PNet

def analyze_video_stream(video_source=0, frame_interval=2, all_frames=False, output_file="positions.json"):
    detector = P2PNet()
    cap = cv2.VideoCapture(video_source)

    if not cap.isOpened():
        print("Error: No se pudo abrir el video.")
        return

    last_processed_time = 0
    frames_data = []

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Fin del video o error en la captura.")
            break

        current_time = time.time()
        if all_frames or current_time - last_processed_time >= frame_interval:
            last_processed_time = current_time

            # Timestamp del frame en segundos (float con 3 decimales)
            timestamp = round(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000, 3)

            # Calcular el tiempo que tarda en realizar la detección
            inicio = time.time_ns()
            frame, points = detector.run(frame)
            final = time.time_ns() - inicio
            print(f"Detection: {final/1000000}ms")

            detector.draw_predictions(frame, points)

            # Guardar posiciones
            frame_info = {
                "timestamp": timestamp,
                "points": [
                    {"id": i, "x": int(x), "y": int(y)} for i, (x, y) in enumerate(points)
                ]
            }
            frames_data.append(frame_info)

            # Mostrar el video en tiempo real
            cv2.imshow("Video Stream", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Guardar datos en JSON

    output = {"source": video_source, "data": frames_data}

    with open(output_file, "w") as f:
        json.dump(output, f, indent=4)
    print(f"Datos guardados en {output_file}")

if __name__ == "__main__":
    analyze_video_stream(video_source="/home/andres/Escritorio/P2PNET_ROOT/was/Whats_004.mp4", frame_interval=1.2, all_frames=True)

from P2PNetDetector import P2PNet
import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

IMG_PATH = "./vis/IMG_2.jpg"


def plot_heatmap(points):
    x, y = zip(*points)
    
    plt.figure(figsize=(8, 6))
    sns.kdeplot(x=x, y=y, cmap='Reds', fill=True, thresh=0)
    plt.scatter(x, y, s=10, c='blue', alpha=0.5, label="Puntos")
    
    plt.title("Mapa de Densidad de Puntos")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.colorbar(label="Densidad")
    plt.show()

def compute_metrics(points):
    x, y = zip(*points)
    metrics = {
        'Total Puntos': len(points),
        'Media X': np.mean(x),
        'Media Y': np.mean(y),
        'Desviación X': np.std(x),
        'Desviación Y': np.std(y)
    }
    return metrics


def main():
    # initialize the P2PNet detector

    detector = P2PNet()
    # load the image
    img = cv2.imread(IMG_PATH)
    # run the detector on the image
    output_frame, points = detector.run(img)

    ########################################## INICIO HOMOGRAFIA
    # Definir puntos de origen en la imagen original (por ejemplo, esquinas de un objeto)
    puntos_origen = np.array([
        [100, 200],  # Esquina superior izquierda
        [400, 180],  # Esquina superior derecha
        [420, 500],  # Esquina inferior derecha
        [80, 520]    # Esquina inferior izquierda
    ], dtype=np.float32)

    # Definir puntos de destino (nueva vista)
    puntos_destino = np.array([
        [150, 200],       # Nuevo punto superior izquierda
        [400, 180],     # Nuevo punto superior derecha
        [420, 500],   # Nuevo punto inferior derecha
        [80, 520]      # Nuevo punto inferior izquierda
    ], dtype=np.float32)

    # Calcular la matriz de transformación homográfica
    M, _ = cv2.findHomography(puntos_origen, puntos_destino)

    output_frame = cv2.warpPerspective(output_frame, M, (img.shape[1], img.shape[0]))
    points = np.array(points)
    points = points.reshape(-1, 1, 2)
    output_points = cv2.perspectiveTransform(points, M)
    output_points = output_points.reshape(-1, 2)
    ########################################## FIN HOMOGRAFIA



    output_frame = detector.draw_predictions(output_frame, output_points) # Esta salida es correcta, por lo que todo se está calculando bien




    output_points = P2PNet.transform_to_plot(output_points, output_frame.shape)

    print(output_points)

    metrics = compute_metrics(output_points)
    plot_heatmap(output_points)
    

    cv2.imshow("Output", output_frame)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__=="__main__":
    main()
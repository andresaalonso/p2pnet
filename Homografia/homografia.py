import cv2
import numpy as np
import matplotlib.pyplot as plt

# Cargar imagen
imagen = cv2.imread("./vis/IMG_2.jpg")  # Cambia por la ruta de tu imagen
imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)  # Convertir de BGR a RGB para Matplotlib

# Definir puntos de origen en la imagen original (por ejemplo, esquinas de un objeto)
puntos_origen = np.array([
    [100, 200],  # Esquina superior izquierda
    [988, 60],  # Esquina superior derecha
    [150, 600],  # Esquina inferior derecha
    [80, 520]    # Esquina inferior izquierda
], dtype=np.float32)

# Definir puntos de destino (nueva vista)
puntos_destino = np.array([
    [150, 200],       # Nuevo punto superior izquierda
    [988, 60],     # Nuevo punto superior derecha
    [150, 600],   # Nuevo punto inferior derecha
    [80, 520]      # Nuevo punto inferior izquierda
], dtype=np.float32)

# Calcular la matriz de transformación homográfica
M, _ = cv2.findHomography(puntos_origen, puntos_destino)

# Aplicar la transformación a la imagen completa
imagen_transformada = cv2.warpPerspective(imagen, M, (imagen.shape[1], imagen.shape[0]))

# Mostrar resultados
plt.figure(figsize=(10, 5))


# Imagen original con los puntos marcados
plt.subplot(1, 2, 1)
plt.imshow(imagen)
plt.scatter(puntos_origen[:, 0], puntos_origen[:, 1], c='red', marker='o', label="Puntos originales")
plt.scatter(puntos_destino[:, 0], puntos_destino[:, 1], c='yellow', marker='o', label="Puntos destino")
plt.legend()
plt.title("Imagen Original")

# Imagen transformada
plt.subplot(1, 2, 2)
plt.imshow(imagen_transformada)
plt.title("Imagen Transformada con Homografía")

plt.show()

import torch
import cv2
import numpy as np
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import os
from matplotlib import pyplot as plt

def get_angle_of_rotation(lines):
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
        angles.append(angle)
    median_angle = np.median(angles)
    return median_angle

def rotate_image(image, angle):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated

# Ocultar la ventana principal de tkinter
Tk().withdraw()

# Solicitar al usuario que seleccione el archivo de imagen
img_path = askopenfilename(title="Seleccione una imagen para la inferencia")

# Verificar que se seleccionó un archivo
if img_path:
    print(f"Archivo seleccionado: {img_path}")

    # Ruta del modelo entrenado
    model_path = "runs/train/exp8/weights/best.pt"

    # Cargar el modelo entrenado
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)

    # Leer la imagen original
    img_original = cv2.imread(img_path)

    # Realizar la primera detección en la imagen original
    results_initial = model(img_path)

    # Convertir resultados a una imagen de OpenCV
    img_cv2 = results_initial.render()[0]

    # Convertir la imagen a escala de grises
    gray = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2GRAY)

    # Aplicar detección de bordes
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # Detectar líneas usando la transformación de Hough
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=50, maxLineGap=10)

    # Calcular el ángulo de rotación
    if lines is not None:
        angle = get_angle_of_rotation(lines)
        print(f"Ángulo de rotación: {angle}")

        # Rotar la imagen
        rotated_img = rotate_image(img_original, angle)

        # Realizar detección en la imagen rotada
        results_rotated = model(rotated_img)
        if len(results_rotated.xyxy[0]) > 0:
            x1, y1, x2, y2, conf, cls = results_rotated.xyxy[0][0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Recortar la imagen usando las coordenadas
            roi = rotated_img[y1:y2, x1:x2]

            # Guardar el recorte con un nombre específico
            base_name = os.path.basename(img_path)
            name, ext = os.path.splitext(base_name)
            output_path = f"recorte_{name}{ext}"
            cv2.imwrite(output_path, roi)
            print(f"Recorte guardado como: {output_path}")

            # Mostrar el recorte usando Matplotlib
            roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            plt.imshow(roi_rgb)
            plt.axis('off')  # Ocultar los ejes
            plt.show()
        else:
            print("No se detectó ninguna etiqueta en la imagen rotada.")
    else:
        print("No se detectaron líneas en la imagen.")
else:
    print("No se seleccionó ningún archivo.")

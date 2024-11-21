import torch
import cv2
import numpy as np
from tkinter import Tk
from tkinter.filedialog import askdirectory
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

# Solicitar al usuario que seleccione la carpeta de imágenes
folder_path = askdirectory(title="Seleccione la carpeta con las imágenes para la inferencia")

# Verificar que se seleccionó una carpeta
if folder_path:
    print(f"Carpeta seleccionada: {folder_path}")

    # Ruta del modelo entrenado
    model_path = "runs/train/exp8/weights/best.pt"

    # Cargar el modelo entrenado
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)

    # Directorio para guardar los recortes
    output_dir = "D:/archivos viejos/test/recortes"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Procesar cada imagen en la carpeta seleccionada
    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)
        print(f"Procesando {img_path}")

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

                # Convertir el recorte a escala de grises para detectar líneas
                roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

                # Aplicar detección de bordes al recorte
                roi_edges = cv2.Canny(roi_gray, 50, 150, apertureSize=3)

                # Detectar líneas en el recorte usando la transformación de Hough
                roi_lines = cv2.HoughLinesP(roi_edges, 1, np.pi / 180, threshold=100, minLineLength=50, maxLineGap=10)

                # Calcular el ángulo de rotación del recorte
                if roi_lines is not None:
                    roi_angle = get_angle_of_rotation(roi_lines)
                    print(f"Ángulo de rotación del recorte: {roi_angle}")

                    # Rotar el recorte
                    roi_rotated = rotate_image(roi, roi_angle)

                    # Guardar el recorte rotado con un nombre específico
                    base_name = os.path.basename(img_path)
                    name, ext = os.path.splitext(base_name)
                    output_path = os.path.join(output_dir, f"recorte_{name}{ext}")
                    cv2.imwrite(output_path, roi_rotated)
                    print(f"Recorte guardado como: {output_path}")

                else:
                    print("No se detectaron líneas en el recorte.")
            else:
                print("No se detectó ninguna etiqueta en la imagen rotada.")
        else:
            print("No se detectaron líneas en la imagen.")
else:
    print("No se seleccionó ninguna carpeta.")

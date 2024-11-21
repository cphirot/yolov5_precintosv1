import torch
from PIL import Image
import cv2
import numpy as np
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import matplotlib.pyplot as plt
import os

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

    # Leer la imagen seleccionada
    img = Image.open(img_path)

    # Realizar inferencia
    results = model(img)

    # Mostrar los resultados usando PIL
    results.show()

    # Convertir la imagen de PIL a OpenCV para visualización adicional
    img_cv2 = np.array(img)

    # Realizar inferencia nuevamente en la imagen OpenCV
    results_cv2 = model(img_cv2)

    # Dibujar los resultados en la imagen
    results_cv2.render()

    # Obtener coordenadas del precinto detectado
    # Asumimos que sólo hay un precinto por imagen, así que tomamos el primer resultado
    if len(results_cv2.xyxy[0]) > 0:
        x1, y1, x2, y2, conf, cls = results_cv2.xyxy[0][0]

        # Convertir las coordenadas a enteros
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        # Recortar la imagen usando las coordenadas
        roi = img_cv2[y1:y2, x1:x2]

        # Convertir BGR a RGB para Matplotlib
        roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)

        # Mostrar la imagen recortada usando Matplotlib
        plt.imshow(roi_rgb)
        plt.axis('off')  # Ocultar los ejes
        plt.show()

        # Guardar el recorte con un nombre específico
        base_name = os.path.basename(img_path)
        name, ext = os.path.splitext(base_name)
        output_path = f"recorte_{name}{ext}"
        cv2.imwrite(output_path, roi)
        print(f"Recorte guardado como: {output_path}")
    else:
        print("No se detectó ningún precinto en la imagen.")
else:
    print("No se seleccionó ningún archivo.")

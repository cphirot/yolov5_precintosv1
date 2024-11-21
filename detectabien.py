import cv2
import numpy as np
import pytesseract

def detect_and_read_seal(image_path):
    # Configurar Tesseract
    pytesseract.pytesseract.tesseract_cmd = r'D:\Software\conda\envs\yolov5-env\Library\bin\tesseract.exe'
    
    # Leer la imagen
    img = cv2.imread(image_path)
    if img is None:
        raise Exception("No se pudo cargar la imagen")

    # Convertir a HSV y crear una máscara para filtrar colores no blancos
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 30, 255])
    mask = cv2.inRange(hsv, lower_white, upper_white)
    
    # Aplicar la máscara a la imagen original
    filtered_img = cv2.bitwise_and(img, img, mask=mask)
    
    # Convertir a escala de grises
    gray = cv2.cvtColor(filtered_img, cv2.COLOR_BGR2GRAY)
    
    # Aplicar umbral para detectar la etiqueta blanca
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    
    # Encontrar contornos
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Agregar un pequeño padding alrededor del ROI
        padding = 10
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(img.shape[1] - x, w + 2*padding)
        h = min(img.shape[0] - y, h + 2*padding)
        
        # Recortar la región de interés
        roi = gray[y:y+h, x:x+w]
        
        # Aumentar el contraste usando CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        roi = clahe.apply(roi)
        
        # Aplicar umbral adaptativo
        thresh = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        
        # Aplicar operaciones morfológicas para limpiar la imagen
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Configuración específica para Tesseract
        custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789'
        
        # Realizar OCR
        text = pytesseract.image_to_string(opening, config=custom_config)
        
        # Mostrar resultados intermedios para debugging
        cv2.imshow('Original', img)
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)  # Dibujar el ROI
        cv2.imshow('Filtered Image', filtered_img)
        cv2.imshow('Detected ROI', img)
        cv2.imshow('ROI', roi)
        cv2.imshow('Threshold', thresh)
        cv2.imshow('Opening', opening)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        # Limpiar el resultado
        number = ''.join(filter(str.isdigit, text))
        
        # Guardar las imágenes procesadas
        cv2.imwrite('debug_filtered_img.jpg', filtered_img)
        cv2.imwrite('debug_roi.jpg', roi)
        cv2.imwrite('debug_threshold.jpg', thresh)
        cv2.imwrite('debug_opening.jpg', opening)
        
        return number
    
    return None

# Ejemplo de uso
ruta_imagen = r'D:\Proyectos\yolov5\yolov5\prueba.jpg'
resultado = detect_and_read_seal(ruta_imagen)
print("Número detectado:", resultado)

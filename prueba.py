import pytesseract
from PIL import Image

# Configurar la ruta de Tesseract
pytesseract.pytesseract.tesseract_cmd = r'D:\Software\conda\pkgs\tesseract-5.3.1-hcb5f61f_0\Library\bin\tesseract.exe'

# Ruta correcta del archivo de imagen
img_path = r'D:\Proyectos\yolov5\yolov5\prueba.jpg'  # Asegúrate de que la ruta sea correcta y sin comillas dobles adicionales

# Cargar la imagen y usar Tesseract para leer el texto
try:
    img = Image.open(img_path)
    text = pytesseract.image_to_string(img, config='--psm 6')
    print(f"Texto reconocido: {text}")
except Exception as e:
    print(f"Error al leer la imagen: {e}")

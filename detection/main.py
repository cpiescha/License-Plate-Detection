import matplotlib.pyplot as plt
import cv2
from ultralytics import YOLO
import pytesseract
import numpy as np

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

ctexto = ''

if __name__ == '__main__':

    cap = cv2.VideoCapture("video_moto.mp4")  # mostrar video 

    # cargar modelo 
    model = YOLO("best_placa.pt")  
    
    while cap.isOpened():
        status, frame = cap.read()

        if not status:
            break
        
        frame = cv2.resize(frame, (900, 800))
        results = model(frame)
        
        # Procesar las detecciones
        for result in results:
            boxes = result.boxes  # Obtener las cajas delimitadoras
            for box in boxes:
                # Obtener las coordenadas de la caja delimitadora
                x1, y1, x2, y2 = map(int, box.xyxy[0].numpy())
                
                # Recortar la región de la placa del frame original
                placa_roi = frame[y1:y2, x1:x2]
                
                
                
                grey= cv2.cvtColor(placa_roi, cv2.COLOR_BGR2GRAY)
                
                _, binario = cv2.threshold(grey,90,255,cv2.THRESH_BINARY)
                
                cv2.imshow('imagen',binario)
                # Aplicar Tesseract para realizar OCR en la región de la placa
                config_placa = '--psm 7 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
                texto = pytesseract.image_to_string(binario, config=config_placa)
                
                ctexto = texto  # Limpiar el texto obtenido
                print(f"Texto detectado: {ctexto[0:7]}")
                
                # Dibujar la caja delimitadora en el frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Mostrar el texto detectado en el frame
                cv2.putText(frame, ctexto[0:7], (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Mostrar el frame con las cajas y el texto detectado
        cv2.imshow("frame", frame)
        
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

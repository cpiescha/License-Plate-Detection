import matplotlib.pyplot as plt
import cv2
from ultralytics import YOLO
import pytesseract
import numpy as np
from PIL import Image,ImageTk
import tkinter as tk
from tkinter import messagebox
from tkinter import filedialog

#pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
cap = None

model = YOLO("best_placa.pt")  

def visualizar():
    global cap
    if cap is not None:
            status, frame = cap.read()

            if status:
            
                frame = cv2.resize(frame, (640, 640))
                results = model(frame)
            
                for result in results:
                    boxes = result.boxes  # Obtener las cajas delimitadoras
                    for box in boxes:
                    # Obtener las coordenadas de la caja delimitadora
                        x1, y1, x2, y2 = map(int, box.xyxy[0].numpy())
                    
                        conf=box.conf[0]
                        placa_roi = frame[y1:y2, x1:x2]
                        
                        
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, f'{conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                        
                            
                    # Recortar la región de la placa del frame original
                    
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame)
                img_tk = ImageTk.PhotoImage(image=img)
                lbl_video.config(image=img_tk)
                lbl_video.img_tk = img_tk  # Evita que la imagen sea recolectada por el garbage collector
                lbl_video.after(10, visualizar)
                show_plate_image(placa_roi)
                
            else:
                lblInfoVideoPath.configure(text="Aún no se ha seleccionado un video")
                lbl_video.image = ""
                cap.release()
                #finalizar()
                
def show_plate_image(plate_img):
    # Convertir la imagen de BGR a RGB para mostrar en Tkinter
    plate_img_rgb = cv2.cvtColor(plate_img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(plate_img_rgb)
    img_tk = ImageTk.PhotoImage(image=img)

    # Actualizar el Label con la imagen recortada de la placa
    lbl_video2.img_tk = img_tk  # Evita que la imagen sea recolectada por el garbage collector
    lbl_video2.config(image=img_tk)
   
def elegir_visualizar_video():
    global cap
    if cap is not None:
        lbl_video.image = ""
        cap.release()
        cap = None
    video_path = filedialog.askopenfilename(filetypes = [
        ("all video format", ".mp4"),
        ("all video format", ".avi")])
    if len(video_path) > 0:
        lblInfoVideoPath.configure(text=video_path)
        cap = cv2.VideoCapture(video_path)
        visualizar()
    else:
        lblInfoVideoPath.configure(text="Aún no se ha seleccionado un video")    
    
root = tk.Tk()
root.title("Reproductor de Video con Detección de Placas")





btnVisualizar = tk.Button(root, text="Elegir y visualizar video", command=elegir_visualizar_video)
btnVisualizar.grid(column=0, row=0, padx=5, pady=5, columnspan=2)

lblInfo1 = tk.Label(root, text="Video de entrada:")
lblInfo1.grid(column=0, row=1)

lblInfoVideoPath = tk.Label(root, text="Aún no se ha seleccionado un video")
lblInfoVideoPath.grid(column=1, row=1)

lbl_video = tk.Label(root)
lbl_video.grid(column=0, row=2, columnspan=2)

lbl_video2=tk.Label(root)
lbl_video2.grid(column=2,row=2,columnspan=2)

root.mainloop()
    
    
    
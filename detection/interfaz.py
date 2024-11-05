import matplotlib.pyplot as plt
import cv2
from ultralytics import YOLO
import pytesseract
import numpy as np
from PIL import Image,ImageTk
import tkinter as tk
from tkinter import messagebox

cap = None
video_running = False
model = YOLO("best_placa.pt")  

def iniciar():
    global cap, video_running
    video_running = True
    cap = cv2.VideoCapture("video_moto.mp4")
    visualizar()


def visualizar():
    global cap,video_running
    if video_running and cap is not None:
            status, frame = cap.read()

            if status:
            
                frame = cv2.resize(frame, (640, 640))
                results = model(frame)
            
                for result in results:
                    boxes = result.boxes  # Obtener las cajas delimitadoras
                    for box in boxes:
                    # Obtener las coordenadas de la caja delimitadora
                        x1, y1, x2, y2 = map(int, box.xyxy[0].numpy())
                    
                        conf=round(float(box.conf[0]),2)
                        placa_roi = frame[y1:y2, x1:x2]
                        
                        
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, f'{conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                        #if conf==0.37:
                            #cv2.imwrite("imagen.jpg",placa_roi)
                            #img=cv2.imread("imagen.jpg")
                        ctext=extract_text_from_plate(placa_roi)
                        if ctext:
                            lbl_plate.config(text=f"Placa detectada: {ctext}")
                            if placa_roi is not None:
                                show_plate_image(placa_roi)  # Mostrar la imagen de la placa recortada
                        else:
                            lbl_plate.config(text="Placa detectada: Ninguna")
                            lbl_plate_img.config(image="")
                    # Recortar la región de la placa del frame original
                    
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame)
                img_tk = ImageTk.PhotoImage(image=img)
                lbl_video.config(image=img_tk)
                lbl_video.img_tk = img_tk  # Evita que la imagen sea recolectada por el garbage collector
                lbl_video.after(10, visualizar)
                show_plate_image(placa_roi)
                
            else:
                finalizar()
                
def show_plate_image(plate_img):
    # Convertir la imagen de BGR a RGB para mostrar en Tkinter
    plate_img_rgb = cv2.cvtColor(plate_img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(plate_img_rgb)
    img_tk = ImageTk.PhotoImage(image=img)

    # Actualizar el Label con la imagen recortada de la placa
    #lbl_video2.img_tk = img_tk  # Evita que la imagen sea recolectada por el garbage collector
    #lbl_video2.config(image=img_tk)
   
    
def finalizar():
    global cap,video_running
    video_running = False
    if cap is not None:
        cap.release()
    lbl_video.image = None
    messagebox.showinfo("Info", "Video finalizado")
    
def extract_text_from_plate(plate_img):
    # Convertir la imagen a escala de grises para mejor resultado en OCR
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    
    #alp,anp,cp=plate_img.shape
    
    
    
   
    # Mva = np.zeros((alp,anp))

    # #                 #normalizamos las matrices

    # nblue= np.matrix(plate_img[:,:,0])
    # ngreen= np.matrix(plate_img[:,:,1])
    # nred= np.matrix(plate_img[:,:,2])

    # #                 #se crea una mascara

    # for col in range(0,alp):
    #     for fil in range(0,anp):
    #         Max= max(nred[col,fil],ngreen[col,fil],nblue[col,fil])
    #         Mva[col,fil] = 255 - Max

    #                 #binarizamos la imagen
    # _, bin = cv2.threshold(Mva,150,255,cv2.THRESH_BINARY)



    #                 #convertimos la matriz en imagen
    # bin = bin.reshape(alp,anp)

    # bin = Image.fromarray(bin)


    # bin = bin.convert("L")
 
    
    # Aplicar algún procesamiento adicional si es necesario (umbral, reducción de ruido, etc.)
    # plate_img = cv2.threshold(gray_plate, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    # Usar pytesseract para extraer texto
    

    # Convertir la imagen a escala de grises
    gray_img = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    
    # Aplicar reducción de ruido con un filtro gaussiano
    blurred_img = cv2.GaussianBlur(gray_img, (5, 5), 0)
    
    # Binarización adaptativa para obtener mejor contraste
    binary_img = cv2.adaptiveThreshold(blurred_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY, 11, 2)
    
    # Aplicar erosión y dilatación para resaltar caracteres
    kernel = np.ones((2, 2), np.uint8)
    processed_img = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, kernel)
    
    # Guardar la imagen preprocesada para depuración (opcional)
    cv2.imwrite('processed_plate.jpg', processed_img)

    # Convertir la imagen a formato PIL para usar con pytesseract
    pil_img = Image.fromarray(processed_img)
    config_placa = '--psm 7 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    text = pytesseract.image_to_string(pil_img, config=config_placa)  # psm 8 para placas (una sola línea de texto)
    
    
    return text.split()
    
root = tk.Tk()
root.title("Reproductor de Video con Detección de Placas")

# Crear un Label para mostrar el video
lbl_video = tk.Label(root)
lbl_video.pack()

# lbl_video2 = tk.Label(root)
# lbl_video2.pack(pady=5)
# lbl_video2.config(width=100,height=30)


# Botones para iniciar y detener el video
btn_start = tk.Button(root, text="Iniciar Video", command=iniciar)
btn_start.pack(side=tk.LEFT, padx=10, pady=10)

btn_stop = tk.Button(root, text="Finalizar Video", command=finalizar)
btn_stop.pack(side=tk.RIGHT, padx=10, pady=10)

lbl_plate = tk.Label(root, text="Placa detectada: Ninguna", font=("Helvetica", 16))
lbl_plate.pack(pady=10)

# Crear un Label para mostrar la imagen de la placa recortada
lbl_plate_img = tk.Label(root)
lbl_plate_img.pack(pady=10)
# Iniciar el loop principal de la ventana
root.mainloop()
    
    
    
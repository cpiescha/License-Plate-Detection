

# ensaando entrenamiento 


import torch
from ultralytics import YOLO

print("CUDA disponible:", torch.cuda.is_available())  # true si esta conectada 
print(torch.__version__)

from roboflow import Roboflow

# crear unacuenta en robofow y el genera el codigo para  descargar el .yml
rf = Roboflow(api_key="HI")


project = rf.workspace("haeun-kim-ri91b").project("license-plate-detection-wienp")
version = project.version(2)
dataset = version.download("yolov8") # descargar yolov





# ejecutar en la terminal para entrenar  =  yolo task=detect mode=train model=yolov8s.pt data=D:/DESAROLLO_NUEVO_/RedesNeuronales/yolov8m/datasets/data.yaml epochs=25 imgsz=640 plots=True

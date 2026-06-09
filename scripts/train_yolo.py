from ultralytics import YOLO
import torch

if __name__ == '__main__':

    model = YOLO('yolov8n.pt') 

 
    model.train(
        data='D:/workspace/DeepLearning/Pj_Drowsiness/data/data.yaml', 
        epochs=50,         
        imgsz=640,         
        batch=16,          
        device=0,          
        workers=4,          
        project='runs/drowsiness', 
        name='yolov8_model'
    )
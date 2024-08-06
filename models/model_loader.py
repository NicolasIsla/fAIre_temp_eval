from ultralytics import YOLO
import torch

def load_model(model_type='yolov8', model_version='dummie_smoke.pt'):
    model_path = f'models/{model_type}/{model_version}.pt'
    
    if model_type == 'yolov8' or model_type == 'yolov5':
        return YOLO(model_path)
    elif model_type == 'detr':
        return torch.load(model_path)
    else:
        raise ValueError("Model not supported")

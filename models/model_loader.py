from ultralytics import YOLO
import torch
from models.lstm_resnet.fire_classifier import FireClassifier
from models.lstm_effnet.fire_classifier import FireClassifierv2

def load_model(model_type='yolov8', model_version='dummie_smoke.pt'):
    model_path = f'models/{model_type}/{model_version}.pt'
    
    if model_type == 'yolov8' or model_type == 'yolov5':
        return YOLO(model_path)
    elif model_type == 'detr':
        return torch.load(model_path)
    elif model_type == 'lstm_resnet':
        return FireClassifier.load_from_checkpoint(f'models/lstm_resnet/{model_version}.ckpt')
    elif model_type == 'lstm_effnet':
        return FireClassifierv2.load_from_checkpoint(f'models/lstm_effnet/{model_version}.ckpt')
    else:
        raise ValueError(f"Model {model_type} not supported")

    
    

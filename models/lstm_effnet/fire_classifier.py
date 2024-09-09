import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet50_Weights
from torchvision import transforms
from PIL import Image
import pytorch_lightning as pl

from timm import create_model

class FireClassifierv2(pl.LightningModule):
    def __init__(self, lstm_layers=4):
        super(FireClassifierv2, self).__init__()
        self.save_hyperparameters()

        self.resize = transforms.Resize((112, 112))
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        # Usamos EfficientNetB3 como extractor de características.
        efficientnet = create_model('efficientnet_b3', pretrained=True)

        # Removemos la capa final de clasificación para usarla como extractor de características.
        self.feature_extractor = nn.Sequential(*list(efficientnet.children())[:-2])

        # Número de características de salida de EfficientNetB3.
        num_features = efficientnet.classifier.in_features  # Cambiado según EfficientNetB3

        # LSTM que procesará las características extraídas.
        self.lstm = nn.LSTM(input_size=num_features * 4 * 4, hidden_size=256, batch_first=True, num_layers=lstm_layers)

        # Capa de clasificación.
        self.classifier = nn.Linear(256, 1)  # Salida binaria

        # Dropout para regularización
        self.dropout = nn.Dropout(0.2)


    def forward(self, x):
        # x shape: [batch_size, seq_length, channels, height, width]
        # Procesa cada imagen de la secuencia a través del extractor de características.
        batch_size, seq_length, C, H, W = x.size()
        x = x.view(batch_size * seq_length, C, H, W)
        x = self.feature_extractor(x)

        # Reformatear salida para la LSTM
        x = x.view(batch_size, seq_length, -1)

        # Pasar las características por la LSTM
        x, _ = self.lstm(x)

        # Aplicamos la capa de clasificación a cada frame de la secuencia
        # x = self.dropout(x)
        x = self.classifier(x)
        return x

    def preprocess_image(self, img):
        img = self.resize(img)
        img = self.to_tensor(img)
        img = self.normalize(img)
        return img

    def infer_4_frames(self, image_paths):
        images = [Image.open(img_path) for img_path in image_paths]
        
        preprocessed_images = [self.preprocess_image(img).unsqueeze(0) for img in images]
        
        input_tensor = torch.cat(preprocessed_images).unsqueeze(0)  # (1, 4, C, H, W)

        self.eval()
        with torch.no_grad():
            output = self(input_tensor)
            pred = torch.sigmoid(output).item()
            return pred
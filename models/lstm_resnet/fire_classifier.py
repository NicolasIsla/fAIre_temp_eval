import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet50_Weights
from torchvision import transforms
from PIL import Image
import pytorch_lightning as pl

class FireClassifier(pl.LightningModule):
    def __init__(self):
        super(FireClassifier, self).__init__()
        resnet = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-2])
        self.lstm = nn.LSTM(input_size=32768, hidden_size=256, batch_first=True, num_layers=1)
        self.classifier = nn.Linear(256, 1)

        self.resize = transforms.Resize((112, 112))
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def forward(self, x):
        batch_size, seq_length, C, H, W = x.size()
        x = x.view(batch_size * seq_length, C, H, W)
        x = self.feature_extractor(x)
        x = x.view(batch_size, seq_length, -1)
        x, _ = self.lstm(x)
        x = x[:, -1, :]
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
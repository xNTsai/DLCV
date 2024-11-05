import torch
import torch.nn as nn
from torchvision import models
from p1_pretrain import pretrain_ssl_model
import os

class FullModel(nn.Module):
    def __init__(self, pretrained_backbone, num_classes):
        super().__init__()
        self.backbone = pretrained_backbone
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.classifier = nn.Sequential(
            nn.Linear(num_features, 1024, bias=False),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.4),
        )
        self.final_layer = nn.Linear(1024, num_classes)

    def forward(self, x):
        features = self.backbone(x)
        features = self.classifier(features)
        return self.final_layer(features)
    
    def get_embedding(self, x):
        features = self.backbone(x)
        features = self.classifier(features)
        return features

def create_model(model_type, num_classes, device):
    if model_type in ['A', 'B', 'C']:
        resnet = models.resnet50(weights=None)
        if model_type == 'B':
            resnet.load_state_dict(torch.load('./hw1_data/p1_data/pretrain_model_SL.pt', map_location=device, weights_only=True))
            # resnet.load_state_dict(torch.load('/kaggle/input/dlcv-hw1-data/hw1_data/p1_data/pretrain_model_SL.pt'))
        elif model_type == 'C':
            ssl_model_path = pretrain_ssl_model(device)  # Call pretraining function
            resnet.load_state_dict(torch.load(ssl_model_path, map_location=device, weights_only=True))

        model = FullModel(resnet, num_classes)
    elif model_type in ['D', 'E']:
        resnet = models.resnet50(weights=None)
        if model_type == 'D':
            resnet.load_state_dict(torch.load('./hw1_data/p1_data/pretrain_model_SL.pt', map_location=device, weights_only=True))
            # resnet.load_state_dict(torch.load('/kaggle/input/dlcv-hw1-data/hw1_data/p1_data/pretrain_model_SL.pt'))
        elif model_type == 'E':
            if os.path.exists('./p1_pretrain_checkpoint/final_pretrain_model.pt'):
                resnet.load_state_dict(torch.load('./p1_pretrain_checkpoint/final_pretrain_model.pt', map_location=device, weights_only=True))
            else:
                ssl_model_path = pretrain_ssl_model(device)  # Call pretraining function
                resnet.load_state_dict(torch.load(ssl_model_path))
        for param in resnet.parameters():
            param.requires_grad = False
        model = FullModel(resnet, num_classes)
    else:
        raise ValueError("Invalid model type. Choose from A, B, C, D, or E.")
    return model
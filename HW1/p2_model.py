import torch
import torch.nn as nn
from torchvision.models import VGG16_Weights, vgg16
from torchvision.models.segmentation.deeplabv3 import DeepLabHead, FCNHead
from torchvision.models.segmentation import (
    DeepLabV3_ResNet50_Weights,
    DeepLabV3_ResNet101_Weights,    # Added import for ResNet101 weights
    DeepLabV3_MobileNet_V3_Large_Weights
)

def DeepLabv3(n_classes=7, mode='resnet'):
    if mode == 'resnet':
        model = torch.hub.load(
            'pytorch/vision:v0.10.0',
            'deeplabv3_resnet50',
            weights=DeepLabV3_ResNet50_Weights.DEFAULT
        )
        model.classifier = DeepLabHead(2048, n_classes)
        model.aux_classifier = FCNHead(1024, n_classes)
    elif mode == 'resnet101':    # Added ResNet101 mode
        model = torch.hub.load(
            'pytorch/vision:v0.10.0',
            'deeplabv3_resnet101',
            weights=DeepLabV3_ResNet101_Weights.DEFAULT
        )
        model.classifier = DeepLabHead(2048, n_classes)
        model.aux_classifier = FCNHead(1024, n_classes)
    elif mode == 'mobile':
        model = torch.hub.load(
            'pytorch/vision:v0.10.0',
            'deeplabv3_mobilenet_v3_large',
            weights=DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT
        )
        model.classifier = DeepLabHead(960, n_classes)
        model.aux_classifier = FCNHead(320, n_classes)
    else:
        raise ValueError(
            f"Unsupported mode: {mode}. Supported modes are 'resnet', 'resnet101', and 'mobile'."
        )

    model.train()
    return model


class FCN32s(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        
        # Load pre-trained VGG16 model
        self.vgg16 = vgg16(weights=VGG16_Weights.DEFAULT)
        
        # Extract features from VGG16
        self.features = self.vgg16.features
        
        # Replace fully connected layers with convolutional layers
        self.fc6 = self._make_conv_layer(512, 2048, 7)
        self.fc7 = self._make_conv_layer(2048, 2048, 1)
        
        # Final scoring layer
        self.score_fr = nn.Conv2d(2048, num_classes, kernel_size=1)
        
        # Upsampling layer
        self.upscore = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=64, stride=32)
        
    def _make_conv_layer(self, in_channels, out_channels, kernel_size):
        conv_layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding='same'),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
        )
        return conv_layer
    
    def forward(self, x):
        input_size = x.size()[2:]
        
        # Extract features
        x = self.features(x)
        
        # Apply fully convolutional layers
        x = self.fc6(x)
        x = self.fc7(x)
        
        # Apply final scoring layer
        x = self.score_fr(x)
        
        # Upsample the output
        x = self.upscore(x)
        
        # Crop the output to match the input size
        x = x[..., :input_size[0], :input_size[1]]
        
        return x
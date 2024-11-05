import argparse
import torch
import torch.nn as nn
import torchvision.transforms as trns
from torchvision import models
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

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

def create_model(num_classes, device):
    resnet = models.resnet50(weights=None)
    model = FullModel(resnet, num_classes)
    return model

class testDataset(Dataset):
    def __init__(self, csv_path, folder_path, transform):
        self.df = pd.read_csv(csv_path)
        self.folder_path = folder_path
        self.transform = transform
        self.num_classes = 65

    def __getitem__(self, index):
        row = self.df.iloc[index]
        img_path = os.path.join(self.folder_path, row['filename'])
        image = Image.open(img_path).convert('RGB')
        return self.transform(image), row['filename']

    def __len__(self):
        return len(self.df)

def inference(csv_path, image_folder_path, output_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Device used: {device}')

    # Load the best model
    best_model_path = f'./hw1_p1_best_model.pt'
    if not os.path.exists(best_model_path):
        raise FileNotFoundError(f"Best model not found at {best_model_path}")

    # Define transforms
    mean = [0.6062, 0.5748, 0.5421]
    std = [0.2398, 0.2409, 0.2457]
    test_transform = trns.Compose([
        trns.Resize((128, 128)),
        trns.ToTensor(),
        trns.Normalize(mean=mean, std=std)
    ])

    # Load test dataset
    test_dataset = testDataset(csv_path, image_folder_path, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)

    # Create and load the model
    model = create_model(test_dataset.num_classes, device)
    model.load_state_dict(torch.load(best_model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    # Read the input CSV to get the correct id and filename order
    df_input = pd.read_csv(csv_path)
    # Create a dictionary to store predictions
    predictions_dict = {}

    # Perform inference
    with torch.no_grad():
        for inputs, files in tqdm(test_loader, desc='Inference'):
            inputs = inputs.to(device)
            outputs = model(inputs)
            y_pred = torch.argmax(outputs, axis=1).flatten().detach().tolist()
            predictions_dict.update(zip(files, y_pred))

    # Write predictions to file
    with open(output_path, 'w') as f:
        f.write("id,filename,label\n")  # Write header
        for _, row in df_input.iterrows():
            row_id, filename = row['id'], row['filename']
            label = predictions_dict.get(filename, 'None')  # Use 'None' if filename not found
            f.write(f"{row_id},{filename},{label}\n")

    print(f"Predictions saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform inference using trained model")
    parser.add_argument("csv_path", type=str, help="Path to the images CSV file")
    parser.add_argument("image_folder_path", type=str, help="Path to the folder containing images")
    parser.add_argument("output_path", type=str, help="Path of output .csv file (predicted labels)")
    args = parser.parse_args()

    inference(args.csv_path, args.image_folder_path, args.output_path)
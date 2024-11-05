import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as trns
from tqdm import tqdm
from p1_officeDataloader import officeDataset
from p1_model import create_model
from torch.amp import GradScaler
from torch.amp import autocast
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from p1_visualization import visualize_tSNE



# Parse command line arguments
parser = argparse.ArgumentParser(description="Train models for Office-Home dataset")
parser.add_argument("-m", "--model", type=str, choices=['A', 'B', 'C', 'D', 'E'],
                    required=True, help="Type of model to train (A, B, C, D, or E)")
args = parser.parse_args()

mean = [0.6062, 0.5748, 0.5421]
std = [0.2398, 0.2409, 0.2457]

# Load Office-Home dataset
office_train_transform = trns.Compose([
    trns.Resize((144, 144)),  # Slightly larger for more aggressive cropping
    trns.RandomCrop(128, padding=4),
    trns.RandomRotation(15),
    trns.RandomHorizontalFlip(),
    trns.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    trns.RandomGrayscale(p=0.2),
    trns.ToTensor(),
    trns.Normalize(mean=mean, std=std)
])

office_val_transform = trns.Compose([
    trns.Resize((128, 128)),
    trns.ToTensor(),
    trns.Normalize(mean=mean, std=std)
])

print('office_home_train loading...')
office_home_train = officeDataset('./hw1_data/p1_data/office/train', transform=office_train_transform)
print('office_home_train loaded successfully')
print('office_home_val loading...')
office_home_val = officeDataset('./hw1_data/p1_data/office/val', transform=office_val_transform)
print('office_home_val loaded successfully')


office_home_train_loader = DataLoader(office_home_train, batch_size=64, shuffle=True, num_workers=4)
office_home_val_loader = DataLoader(office_home_val, batch_size=64, shuffle=False, num_workers=4)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Device used: {device}')

full_model = create_model(args.model, office_home_train.num_classes, device).to(device)

if args.model in ['D', 'E']:
    optimizer = optim.AdamW(list(full_model.classifier.parameters()) + list(full_model.final_layer.parameters()), lr=1e-3, weight_decay=5e-2)
else:
    optimizer = optim.AdamW(full_model.parameters(), lr=1e-3, weight_decay=5e-2)

num_epochs = 300
plot_epochs = [1, num_epochs]
ckpt_path = f'./p1_checkpoint/{args.model}_train_checkpoint'
os.makedirs(ckpt_path, exist_ok=True)

scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
scaler = GradScaler("cuda" if torch.cuda.is_available() else "cpu")

criterion = nn.CrossEntropyLoss()

final_acc = 0.0

for epoch in range(1, num_epochs + 1):
    full_model.train()
    train_loss = 0.0
    for inputs, labels in tqdm(office_home_train_loader, desc=f'Epoch {epoch} Training'):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        with autocast("cuda" if torch.cuda.is_available() else "cpu"):
            outputs = full_model(inputs)
            loss = criterion(outputs, labels)
        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(full_model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        train_loss += loss.item()
    
    train_loss /= len(office_home_train_loader)
    
    full_model.eval()

    with torch.no_grad():
        va_loss = 0
        correct = 0
        total = 0
        for inputs, labels in tqdm(office_home_val_loader, desc=f'Epoch {epoch} Validation'):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = full_model(inputs)
            va_loss += criterion(outputs, labels).item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        va_loss /= len(office_home_val_loader)
        va_acc = 100 * correct / total
        
        print(f"epoch {epoch}, va_acc = {va_acc:.2f}%, va_loss = {va_loss:.4f}")
        if va_acc > final_acc:
            final_acc = va_acc
            torch.save(optimizer.state_dict(), os.path.join(ckpt_path, 'final_optimizer.pt'))
            torch.save(full_model.state_dict(), os.path.join(ckpt_path, 'final_model.pt'))
            print("New model saved successfully!")

    scheduler.step()

    if epoch in plot_epochs:
        print(f"Creating t-SNE visualization for epoch {epoch}...")
        visualize_tSNE(full_model, office_home_train_loader, device, epoch, ckpt_path, args.model, train=True)
        visualize_tSNE(full_model, office_home_val_loader, device, epoch, ckpt_path, args.model, train=False)


print(f'Final Validation Accuracy: {final_acc:.2f}%')

# Write final validation accuracy
with open(f'{ckpt_path}/final_val_acc.txt', 'w') as f:
    f.write(f'Final Validation Accuracy: {final_acc:.2f}%')
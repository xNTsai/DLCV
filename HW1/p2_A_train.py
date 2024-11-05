import os
import numpy as np
import torch
import torchvision.transforms as transforms
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.optim import AdamW
from mean_iou_evaluate import mean_iou_score

from p2_dataloader import P2Dataset
from p2_model import FCN32s

def train(net, train_loader, criterion, optimizer, device):
    net.train()
    train_loss = 0.0
    for images, labels in tqdm(train_loader, desc="Training"):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = net(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    return train_loss / len(train_loader)

def validate(net, valid_loader, criterion, device):
    net.eval()
    valid_loss = 0.0
    accuracies = []
    predictions_list = []
    labels_list = []
    with torch.no_grad():
        for images, labels in tqdm(valid_loader, desc="Validation"):
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            loss = criterion(outputs, labels)
            valid_loss += loss.item()
            predictions = outputs.argmax(dim=1)
            accuracy = (predictions == labels).float().mean()
            accuracies.append(accuracy.item())
            
            # Store predictions and labels for mIoU calculation
            predictions_list.extend(predictions.cpu().numpy())
            labels_list.extend(labels.cpu().numpy())

    # Calculate mIoU
    miou = mean_iou_score(np.array(predictions_list), np.array(labels_list))
    
    return valid_loss / len(valid_loader), sum(accuracies) / len(accuracies), miou

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Define hyperparameters
    epochs = 100
    batch_size = 8
    initial_lr = 1e-4
    weight_decay = 5e-4  # Updated weight decay
    T_0 = 10  # Number of epochs before first restart
    T_mult = 2  # Factor to increase T_0 after each restart
    patience = 30  
    best_miou = 0.0
    ckpt_path = "./P2_A_checkpoint"

    # Create checkpoint directory if it doesn't exist
    os.makedirs(ckpt_path, exist_ok=True)

    # Define data transforms
    mean = [0.485, 0.456, 0.406]  # ImageNet mean
    std = [0.229, 0.224, 0.225]  # ImageNet std
    data_trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    # Advanced data augmentation
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    # Load datasets
    train_dataset = P2Dataset("./hw1_data/p2_data/train", transform=train_transform, train=True)
    valid_dataset = P2Dataset("./hw1_data/p2_data/validation", transform=data_trans, train=True)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Initialize model
    net = FCN32s()
    net = net.to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1, ignore_index=6)
    optimizer = AdamW(net.parameters(), lr=initial_lr, weight_decay=weight_decay)

    # Define learning rate scheduler
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)

    # Training loop
    epochs_without_improvement = 0
    for epoch in range(1, epochs + 1):
        train_loss = train(net, train_loader, criterion, optimizer, device)
        valid_loss, valid_acc, valid_miou = validate(net, valid_loader, criterion, device)

        print(f"Epoch [{epoch}/{epochs}], Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}, Valid Acc: {valid_acc:.4f}, Valid mIoU: {valid_miou:.4f}")

        # Update learning rate scheduler
        scheduler.step()

        if valid_miou > best_miou:
            best_miou = valid_miou
            torch.save(optimizer.state_dict(), os.path.join(ckpt_path, "best_optimizer.pth"))
            torch.save(net.state_dict(), os.path.join(ckpt_path, "best_checkpoint.pt"))
            print("New best model saved!")
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        # Early stopping
        if epochs_without_improvement >= patience:
            print(f"No improvement for {patience} epochs. Early stopping.")
            print(f"Best mIoU: {best_miou:.4f}")
            break

if __name__ == "__main__":
    main()
import os
import numpy as np
import torch
import torchvision.transforms as transforms
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.optim.lr_scheduler import OneCycleLR
from torch.nn import functional as F

from mean_iou_evaluate import mean_iou_score
from p2_model import DeepLabv3
from p2_dataloader import P2Dataset

# Remove FocalLoss class

def train(net, train_loader, criterion, optimizer, scheduler, device):
    net.train()
    total_loss = 0
    for images, labels in tqdm(train_loader, desc="Training"):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = net(images)
        logits, aux_logits = outputs['out'], outputs['aux']
        # logits, aux_logits = outputs['out'], outputs.get('aux', None)
        loss = criterion(logits, labels) + criterion(aux_logits, labels)
        
        loss.backward()
        optimizer.step()
        scheduler.step()  # Move scheduler.step() here

        total_loss += loss.item()
    
    return total_loss / len(train_loader)

def validate(net, valid_loader, criterion, device):
    net.eval()
    valid_loss = 0.0
    preds_list = []
    labels_list = []
    with torch.no_grad():
        for images, labels in tqdm(valid_loader, desc="Validation"):
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            loss = criterion(outputs['out'], labels)
            valid_loss += loss.item()
            preds = outputs['out'].argmax(dim=1).cpu().numpy()
            labels = labels.cpu().numpy()
            preds_list.append(preds)
            labels_list.append(labels)
    preds_list = np.concatenate(preds_list)
    labels_list = np.concatenate(labels_list)
    miou = mean_iou_score(preds_list, labels_list)
    return valid_loss / len(valid_loader), miou

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    epochs = 100
    batch_size = 8
    initial_lr = 1e-4
    weight_decay = 5e-4  # Updated weight decay
    T_0 = 10  # Number of epochs before first restart
    T_mult = 2  # Factor to increase T_0 after each restart
    patience = 100
    best_miou = 0.0
    ckpt_path = "./P2_B_checkpoint"

    saved_epoch = [1, epochs//2, epochs]

    os.makedirs(ckpt_path, exist_ok=True)

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    train_dataset = P2Dataset("hw1_data/p2_data/train", transform=train_transform, train=True)
    valid_dataset = P2Dataset("hw1_data/p2_data/validation", transform=valid_transform, train=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    net = DeepLabv3(n_classes=7, mode='resnet')
    net = net.to(device)

    # class_weights = torch.tensor([0.25, 0.25, 0.5, 0.25, 0.25, 0.3, 0.25]).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1, ignore_index=6)
    
    optimizer = torch.optim.SGD(
        net.parameters(),
        lr=1e-2,
        momentum=0.85,
        weight_decay=weight_decay
    )
    
    # OneCycleLR scheduler
    scheduler = OneCycleLR(
        optimizer,
        max_lr=2e-2,
        steps_per_epoch=len(train_loader),
        epochs=epochs  # Using the existing 'epochs' variable
    )

    epochs_without_improvement = 0
    for epoch in range(1, epochs + 1):
        train_loss = train(net, train_loader, criterion, optimizer, scheduler, device)
        valid_loss, valid_miou = validate(net, valid_loader, criterion, device)
        
        print(f"Epoch [{epoch}/{epochs}], Train Loss: {train_loss:.4f}, "
              f"Valid Loss: {valid_loss:.4f}, Valid mIoU: {valid_miou:.4f}")

        if valid_miou > best_miou:
            best_miou = valid_miou
            torch.save(optimizer.state_dict(), os.path.join(ckpt_path, "best_optimizer.pth"))
            torch.save(net.state_dict(), os.path.join(ckpt_path, "best_model.pth"))
            print("New best model saved!")
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(f"No improvement for {patience} epochs. Early stopping.")
            break

        if epoch in saved_epoch:
            torch.save(net.state_dict(), os.path.join(ckpt_path, f"model_epoch_{epoch}.pth"))
            print(f"Model saved at epoch {epoch}")

    print(f"Best mIoU: {best_miou:.4f}")

if __name__ == "__main__":
    main()
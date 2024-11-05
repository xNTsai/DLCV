import os
import torch
import torchvision.transforms as trns
from torch.utils.data import DataLoader
from torchvision import models
from tqdm import tqdm
from byol_pytorch import BYOL
from p1_miniDataloader import miniDataset
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import OneCycleLR

def pretrain_ssl_model(device, epochs=200, batch_size=128):
    train_transform = trns.Compose([
        trns.Resize((144, 144)),
        trns.RandomCrop(128, padding=4),
        trns.RandomHorizontalFlip(),
        trns.RandomRotation(15),
        trns.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        trns.RandomGrayscale(p=0.2),
        trns.RandomApply([trns.GaussianBlur(kernel_size=3)], p=0.3),
        trns.ToTensor(),
        trns.RandomErasing(p=0.2),
        trns.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = miniDataset(
        './hw1_data/p1_data/mini/train',
        transform=train_transform,
    )
    print('mini_train loading...')
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    print('mini_train loaded successfully')
    ckpt_path = './p1_pretrain_checkpoint'
    if not os.path.isdir(ckpt_path):
        os.mkdir(ckpt_path)

    resnet = models.resnet50(weights=None)
    learner = BYOL(
        resnet,
        image_size=128,
        hidden_layer='avgpool',
        projection_size=256,  # Reduce projection size
        projection_hidden_size=2048,  # Reduce projection hidden size
        moving_average_decay=0.996 
    )
    learner = learner.to(device)
    learner.train()

    opt = torch.optim.AdamW(learner.parameters(), lr=3e-4, weight_decay=0.1)
    scheduler = OneCycleLR(opt, max_lr=1e-3, epochs=epochs, steps_per_epoch=len(train_loader), pct_start=0.1)

    scaler = GradScaler("cuda" if torch.cuda.is_available() else "cpu")

    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        for image in tqdm(train_loader, desc=f'Epoch {epoch}'):
            image = image.to(device)
            opt.zero_grad()
            
            with autocast("cuda" if torch.cuda.is_available() else "cpu"):
                loss = learner(image)
            
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(learner.parameters(), max_norm=1.0)
            scaler.step(opt)
            scaler.update()
            
            learner.update_moving_average()
            running_loss += loss.item()
            scheduler.step()

        epoch_loss = running_loss / len(train_loader)
        print(f"epoch {epoch}, loss = {epoch_loss}")

    # save your final network
    final_model_path = os.path.join(ckpt_path, 'final_pretrained_model.pt')
    torch.save(resnet.state_dict(), final_model_path)
    return final_model_path


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Device used:', device)
    pretrain_ssl_model(device)
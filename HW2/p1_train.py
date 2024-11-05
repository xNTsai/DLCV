import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from pathlib import Path
import torch
from torch.utils.data import DataLoader, ConcatDataset
# from torch.utils.tensorboard.writer import SummaryWriter
from torchvision import transforms
# from torchvision.utils import make_grid
from tqdm import tqdm
import pandas as pd
from PIL import Image

from p1_model import DDPM_framework, Unet

def rm_tree(pth: Path):
    if pth.is_dir():
        for child in pth.iterdir():
            if child.is_file():
                child.unlink()
            else:
                rm_tree(child)
        pth.rmdir()

class CustomDigitDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, csv_file, transform=None, dataset_label=0):
        self.root_dir = Path(root_dir)
        self.csv_file = pd.read_csv(csv_file)
        self.transform = transform
        self.dataset_label = dataset_label

    def __len__(self):
        return len(self.csv_file)

    def __getitem__(self, idx):
        img_name = self.csv_file.iloc[idx, 0]
        img_path = self.root_dir / f"{img_name}"
        image = Image.open(img_path).convert('RGB')
        digit_label = self.csv_file.iloc[idx, 1]

        if self.transform:
            image = self.transform(image)

        return image, digit_label, self.dataset_label

batch_size = 256
num_epochs = 200  # Increase number of epochs
n_T = 500
lr = 1e-4  # Decrease learning rate
n_features = 128  # Increase number of features in Unet
ckpt_path = Path('./P2_ckpt')
# tb_path = Path('./P2_tb')

rm_tree(ckpt_path)
# rm_tree(tb_path)

ckpt_path.mkdir(exist_ok=True)
# tb_path.mkdir(exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Data loading
transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

mnist_m = CustomDigitDataset(
    root_dir='hw2_data/digits/mnistm/data/',
    csv_file='hw2_data/digits/mnistm/train.csv',
    transform=transform,
    dataset_label=0
)

svhn = CustomDigitDataset(
    root_dir='hw2_data/digits/svhn/data/',
    csv_file='hw2_data/digits/svhn/train.csv',
    transform=transform,
    dataset_label=1
)

# Combine datasets
combined_dataset = ConcatDataset([mnist_m, svhn])
train_loader = DataLoader(combined_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

ddpm = DDPM_framework(
    network=Unet(
        in_chans=3,
        n_features=n_features,
        n_classes=10,
        n_datasets=2
    ),
    betas=(1e-4, 0.02),
    n_T=n_T,
    device=device,
    drop_prob=0.1
).to(device)

# Replace Adam with Adam
optim = torch.optim.Adam(ddpm.parameters(), lr=lr)

# Use CosineAnnealingLR instead of OneCycleLR
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=num_epochs, eta_min=lr/20)

scaler = torch.amp.GradScaler('cuda')
# writer = SummaryWriter(tb_path)

for epoch in range(num_epochs):
    ddpm.train()

    epoch_loss = 0.0
    num_batches = 0

    for x, c, d in tqdm(train_loader):
        with torch.autocast(device_type='cuda' if device != 'cpu' else 'cpu', dtype=torch.float16):
            x = x.to(device, non_blocking=True)
            c = c.to(device, non_blocking=True)
            d = d.to(device, non_blocking=True)
            loss = ddpm(x, c, d)
        scaler.scale(loss).backward()
        scaler.step(optim)
        scaler.update()
        optim.zero_grad()

        epoch_loss += loss.item()
        num_batches += 1
    
    # Move scheduler step outside the training loop
    scheduler.step()
    
    avg_loss = epoch_loss / num_batches
    print(f"Epoch {epoch+1}/{num_epochs}, lr: {scheduler.get_last_lr()[0]:.4f}, Average Loss: {avg_loss:.4f}")
    # writer.add_scalar('Training Loss', avg_loss, epoch)

    # ddpm.eval()
    # with torch.no_grad():
    #     n_samples = 30
    #     for gw in [0, 0.5, 2]:
    #         x_gen, x_gen_store = ddpm.sample(n_samples, (3, 28, 28), device, guide_w=gw)
            # grid = make_grid(x_gen * -1 + 1, nrow=3)
            # writer.add_image(f'DDPM results/w={gw:.1f}', grid, epoch)
            # grid = make_grid(x_gen, nrow=3)
            # writer.add_image(f'DDPM results wo inv/w={gw:.1f}', grid, epoch)

    if (epoch+1) % 10 == 0:  # Save checkpoints every 10 epochs
        torch.save(ddpm.state_dict(), ckpt_path / f"{epoch+1}_ddpm.pth")
        print(f"Checkpoint saved at {epoch+1} epoch")

    # current_lr = scheduler.get_last_lr()[0]
    # writer.add_scalar('Learning Rate', current_lr, epoch)

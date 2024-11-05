import random
import sys
from pathlib import Path

import torch
from torchvision.utils import save_image

from p1_model import DDPM_framework, Unet

random.seed(0)
torch.manual_seed(0)

out_dir = Path(sys.argv[1])
out_dir_mnistm = out_dir / 'mnistm'
out_dir_svhn = out_dir / 'svhn'
out_dir_mnistm.mkdir(exist_ok=True, parents=True)
out_dir_svhn.mkdir(exist_ok=True, parents=True)

ckpt_dir = Path('./DLCV_HW2_p1_ckpt.pth')  # Adjust this to your checkpoint path

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = DDPM_framework(
    network=Unet(
        in_chans=3,
        n_features=128,
        n_classes=10,
        n_datasets=2
    ),
    betas=(1e-4, 0.02),
    n_T=500,
    device=device,
    drop_prob=0.1
).to(device)
model.load_state_dict(torch.load(ckpt_dir, map_location=device))
model.eval()

def generate_images(model, n_samples, size, device, digit_class, dataset_label, guide_w=2.0):
    with torch.no_grad():
        x_i, _ = model.class_gen(n_samples, size, device, digit_class, dataset_label, guide_w)
    
    return x_i

print("Generating images...")
for dataset_label, out_dir in enumerate([out_dir_mnistm, out_dir_svhn]):
    for digit_class in range(10):
        print(f"Generating images for digit class {digit_class} in dataset {'MNIST-M' if dataset_label == 0 else 'SVHN'} ...")
        images = generate_images(model, 50, (3, 28, 28), device, digit_class, dataset_label)
        for i, image in enumerate(images, 1):
            save_image(image, out_dir / f'{digit_class}_{i:03d}.png')

print("Image generation complete.")
import torch
import torch.nn as nn
import torchvision.utils as vutils
from UNet import UNet
from utils import beta_scheduler
import os
from PIL import Image
import numpy as np
import argparse

class DDIM:
    def __init__(self, model, n_timestep=1000, device='cuda'):
        self.model = model.to(device)
        self.n_timestep = n_timestep
        self.device = device
        self.betas = beta_scheduler(n_timestep).to(device)
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

    def sample(self, noise, n_steps=50, eta=0):
        x = noise
        timesteps = torch.linspace(self.n_timestep - 1, 0, n_steps + 1).long().to(self.device)
        
        for i in range(n_steps):
            t = timesteps[i]
            t_next = timesteps[i + 1]
            
            alpha_cumprod_t = self.alphas_cumprod[t]
            alpha_cumprod_t_next = self.alphas_cumprod[t_next] if t_next >= 0 else torch.tensor(1.0).to(self.device)

            et = self.model(x, t.expand(noise.shape[0]))
            
            x0_t = (x - et * (1 - alpha_cumprod_t).sqrt()) / alpha_cumprod_t.sqrt()
            c1 = eta * ((1 - alpha_cumprod_t / alpha_cumprod_t_next) * (1 - alpha_cumprod_t_next) / (1 - alpha_cumprod_t)).sqrt()
            c2 = ((1 - alpha_cumprod_t_next) - c1 ** 2).sqrt()
            
            x = alpha_cumprod_t_next.sqrt() * x0_t + c2 * et
            
            if eta > 0:
                noise = torch.randn_like(x)
                x += c1 * noise
        
        return x

def generate_images(model_path, noise_dir, output_dir, n_steps=50, eta=0):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = UNet()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    ddim = DDIM(model, device=device)
    
    os.makedirs(output_dir, exist_ok=True)
    
    for i in range(10):
        noise_path = os.path.join(noise_dir, f'{i:02d}.pt')
        noise = torch.load(noise_path, map_location=device)
        
        with torch.no_grad():
            generated_image = ddim.sample(noise, n_steps=n_steps, eta=eta)
            minVal = generated_image.min()
            maxVal = generated_image.max()
            if minVal < maxVal:
                generated_image = (generated_image - minVal) / (maxVal - minVal)

        # Normalize and save the generated image
        output_path = os.path.join(output_dir, f'{i:02d}.png')
        vutils.save_image(generated_image, output_path)

def calculate_mse(generated_dir, gt_dir):
    mse_scores = []
    
    for i in range(10):
        generated_path = os.path.join(generated_dir, f'{i:02d}.png')
        gt_path = os.path.join(gt_dir, f'{i:02d}.png')
        
        # Compute MSE between generated and ground truth images
        generated_img = Image.open(generated_path).convert('RGB')
        gt_img = Image.open(gt_path).convert('RGB')
        
        # Convert images to numpy arrays
        generated_np = np.array(generated_img).astype(np.float32)
        gt_np = np.array(gt_img).astype(np.float32)
        
        # Calculate MSE
        mse = np.mean((generated_np - gt_np) ** 2)
        mse_scores.append(mse)
        print(f"MSE for image {i}: {mse:.4f}")
    
    avg_mse = np.mean(mse_scores)
    print(f"Average MSE: {avg_mse:.4f}")
    
    return avg_mse

if __name__ == '__main__':
    # model_path = 'hw2_data/face/UNet.pt'
    # noise_dir = 'hw2_data/face/noise'
    # gt_dir = 'hw2_data/face/GT'
    # output_dir = 'Output_folder'

    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', type=str)
    parser.add_argument('noise_dir', type=str)
    parser.add_argument('output_dir', type=str)
    args = parser.parse_args()
    
    generate_images(args.model_path, args.noise_dir, args.output_dir, n_steps=50, eta=0)
    # avg_mse = calculate_mse(args.output_dir, gt_dir)
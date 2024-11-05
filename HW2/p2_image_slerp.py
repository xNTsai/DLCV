import torch
import torchvision.utils as vutils
import numpy as np
from p2_inference import DDIM
from UNet import UNet
import os

def slerp(val, low, high):
    """Spherical linear interpolation."""
    omega = torch.acos(torch.dot(low / torch.norm(low), high / torch.norm(high)))
    so = torch.sin(omega)
    if so == 0:
        return (1.0 - val) * low + val * high  # LERP
    return torch.sin((1.0 - val) * omega) / so * low + torch.sin(val * omega) / so * high

def lerp(val, low, high):
    """Linear interpolation."""
    return (1.0 - val) * low + val * high

def generate_interpolated_images(ddim, noise_00, noise_01, output_dir, n_steps=50):
    alphas = np.linspace(0.0, 1.0, 10)
    
    os.makedirs(output_dir, exist_ok=True)
    
    slerp_images = []
    lerp_images = []
    
    for alpha in alphas:
        # Spherical Linear Interpolation
        slerp_noise = slerp(alpha, noise_00.flatten(), noise_01.flatten()).reshape(noise_00.shape)
        with torch.no_grad():
            slerp_image = ddim.sample(slerp_noise, n_steps=n_steps)
            slerp_image = (slerp_image - slerp_image.min()) / (slerp_image.max() - slerp_image.min())
        slerp_images.append(slerp_image)
        
        # Linear Interpolation
        lerp_noise = lerp(alpha, noise_00.flatten(), noise_01.flatten()).reshape(noise_00.shape)
        with torch.no_grad():
            lerp_image = ddim.sample(lerp_noise, n_steps=n_steps)
            lerp_image = (lerp_image - lerp_image.min()) / (lerp_image.max() - lerp_image.min())
        lerp_images.append(lerp_image)
    
    # Combine images into grids
    slerp_grid = torch.cat(slerp_images, dim=0)
    lerp_grid = torch.cat(lerp_images, dim=0)
    
    # Save grid images
    vutils.save_image(slerp_grid, f"{output_dir}/slerp_grid.png", nrow=10)
    vutils.save_image(lerp_grid, f"{output_dir}/lerp_grid.png", nrow=10)

if __name__ == '__main__':
    model_path = 'hw2_data/face/UNet.pt'
    noise_dir = 'hw2_data/face/noise'
    output_dir = 'Output_folder'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = UNet()
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    
    ddim = DDIM(model, device=device)

    # Load the two noise tensors
    noise_00 = torch.load(f'{noise_dir}/00.pt', map_location=device, weights_only=True)
    noise_01 = torch.load(f'{noise_dir}/01.pt', map_location=device, weights_only=True)

    generate_interpolated_images(ddim, noise_00, noise_01, output_dir)
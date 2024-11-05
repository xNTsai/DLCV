import torch
import torchvision.utils as vutils
from UNet import UNet
from utils import beta_scheduler
import os

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

def generate_grid_image(model_path, noise_dir, output_path, n_steps=50, etas=[0, 0.25, 0.5, 0.75, 1]):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = UNet()
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    
    ddim = DDIM(model, device=device)
    
    grid_images = []
    
    # Load all noise tensors first
    noise_tensors = []
    for i in range(4):
        noise_path = os.path.join(noise_dir, f'{i:02d}.pt')
        noise = torch.load(noise_path, map_location=device, weights_only=True)
        noise_tensors.append(noise)
    
    for eta in etas:
        print(f"Generating images with eta = {eta}")
        row_images = []
        for noise in noise_tensors:
            with torch.no_grad():
                generated_image = ddim.sample(noise, n_steps=n_steps, eta=eta)
                minVal = generated_image.min()
                maxVal = generated_image.max()
                if minVal < maxVal:
                    generated_image = (generated_image - minVal) / (maxVal - minVal)
            
            # Add black padding to the right and bottom of each image
            padded_image = torch.nn.functional.pad(generated_image, (0, 2, 0, 2), value=0)
            row_images.append(padded_image)
        
        # Concatenate images in the row and add black padding at the bottom
        row = torch.cat(row_images, dim=-1)
        row_padded = torch.nn.functional.pad(row, (0, 0, 0, 2), value=0)
        grid_images.append(row_padded)
    
    grid = torch.cat(grid_images, dim=-2)
    
    # Remove the extra padding from the right and bottom edges of the entire grid
    grid = grid[:, :, :-2, :-2]
    
    vutils.save_image(grid, output_path, nrow=1, padding=0)  # Set nrow to 1 to avoid automatic padding

if __name__ == '__main__':
    model_path = 'hw2_data/face/UNet.pt'
    noise_dir = 'hw2_data/face/noise'
    output_path = 'p2_grid_image.png'
    
    generate_grid_image(model_path, noise_dir, output_path, n_steps=50, etas=[0, 0.25, 0.5, 0.75, 1])

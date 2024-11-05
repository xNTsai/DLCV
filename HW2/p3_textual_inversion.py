import os
import json
import torch
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from omegaconf import OmegaConf
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from ldm.util import instantiate_from_config
from einops import rearrange
from ldm.models.diffusion.dpm_solver import DPMSolverSampler


class TextualInversionDataset(Dataset):
    def __init__(self, data_root, size=512, repeats=100):
        self.size = size
        self.repeats = repeats

        data_root = Path(data_root)
        self.image_paths = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            self.image_paths.extend(list(data_root.glob(ext)))

        if not self.image_paths:
            raise ValueError(f"No images found in {data_root}")

        self.transform = transforms.Compose([
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    def __len__(self):
        return len(self.image_paths) * self.repeats

    def __getitem__(self, idx):
        idx = idx % len(self.image_paths)
        image = Image.open(self.image_paths[idx]).convert('RGB')
        image = self.transform(image)
        return image


def load_model_from_config(config, ckpt):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"Missing keys: {missing}")
    if unexpected:
        print(f"Unexpected keys: {unexpected}")
    model.cuda()
    model.eval()
    return model


def train_textual_inversion(args, source_idx, source_data, config, model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    source_map = {
        "dog": "0",
        "David Revoy": "1"
    }

    source_dir = source_map.get(source_data["src_image"])
    if source_dir is None:
        raise ValueError(f"Unknown source image: {source_data['src_image']}")

    source_path = os.path.join(args.input_dir, source_dir)

    dataset = TextualInversionDataset(source_path)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    token_name = source_data["token_name"]  # E.g., "<new1>"
    placeholder_token = token_name
    initializer_token = "art"  # You can choose an appropriate initializer token

    # Add the placeholder token to tokenizer and resize token embeddings
    tokenizer = model.cond_stage_model.tokenizer
    num_added_tokens = tokenizer.add_tokens([placeholder_token])
    if num_added_tokens == 0:
        print(f"Token {placeholder_token} already exists in tokenizer")
    token_ids = tokenizer.convert_tokens_to_ids([placeholder_token])
    model.cond_stage_model.transformer.resize_token_embeddings(len(tokenizer))

    # Initialize the embeddings for the new token
    text_encoder = model.cond_stage_model.transformer
    embeddings = text_encoder.get_input_embeddings()

    # Initialize the new token embedding
    initializer_token_id = tokenizer.convert_tokens_to_ids(initializer_token)
    initializer_embedding = embeddings.weight[initializer_token_id].clone().detach()

    # Create a parameter for the new token's embedding
    new_embedding = torch.nn.Parameter(initializer_embedding.clone().detach())
    new_embedding = new_embedding.to(device)

    # Now, define a function to replace the embedding layer's forward method
    def embedding_forward(input_ids, **kwargs):
        inputs_embeds = embeddings_old(input_ids)
        mask = input_ids == token_ids[0]
        if mask.any():
            inputs_embeds[mask] = new_embedding
        return inputs_embeds

    # Replace the embedding layer's forward method
    embeddings_old = embeddings.forward
    embeddings.forward = embedding_forward

    # Set up optimizer
    optimizer = torch.optim.Adam([new_embedding], lr=5e-4)

    model.train()
    num_epochs = 5  # You can adjust the number of epochs
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in tqdm(dataloader, desc=f"Training Epoch {epoch+1}/{num_epochs}"):
            optimizer.zero_grad()
            batch = batch.to(device)

            # Encode images to latents
            with torch.no_grad():
                latents = model.get_first_stage_encoding(model.encode_first_stage(batch))

            # Get conditioning
            c = model.get_learned_conditioning([f"a photo of {placeholder_token}"])

            # Sample random timesteps
            t = torch.randint(0, model.num_timesteps, (latents.shape[0],), device=device).long()

            # Add noise to the latents
            noise = torch.randn_like(latents)
            noisy_latents = model.q_sample(latents, t, noise=noise)

            # Predict the noise
            noise_pred = model.apply_model(noisy_latents, t, c)

            # Compute loss
            loss = F.mse_loss(noise_pred, noise)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}, Average Loss: {avg_loss:.4f}")

    # After training, restore the original embedding forward method
    embeddings.forward = embeddings_old

    # Update the embedding layer with the trained embedding
    with torch.no_grad():
        embeddings.weight[token_ids[0]] = new_embedding

    # Save the learned embedding
    learned_embeds = {
        "string_to_param": {placeholder_token: new_embedding.detach().cpu()},
        "name": placeholder_token,
        "string_to_token": {placeholder_token: token_ids[0]}
    }

    save_path = os.path.join(args.output_dir, source_idx, f"{placeholder_token}.pt")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(learned_embeds, save_path)
    return save_path


def generate_images(prompt, output_path, model, num_images=25, steps=50):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(output_path, exist_ok=True)

    sampler = DPMSolverSampler(model)
    batch_size = 1  # Adjust based on GPU memory
    C = 4  # Latent channels
    H = W = 512  # Image size
    f = 8  # Downsampling factor

    with torch.no_grad():
        for i in tqdm(range(num_images), desc="Generating Images"):
            uc = model.get_learned_conditioning(batch_size * [""])
            c = model.get_learned_conditioning(batch_size * [prompt])
            shape = [C, H // f, W // f]
            samples, _ = sampler.sample(S=steps,
                                        conditioning=c,
                                        batch_size=batch_size,
                                        shape=shape,
                                        verbose=False,
                                        unconditional_guidance_scale=7.5,
                                        unconditional_conditioning=uc,
                                        eta=0.0)
            x_samples = model.decode_first_stage(samples)
            x_samples = (x_samples + 1.0) / 2.0
            x_samples = torch.clamp(x_samples, 0.0, 1.0)

            # Save images
            for x_sample in x_samples:
                x_sample = x_sample.cpu().numpy()
                x_sample = rearrange(x_sample, 'c h w -> h w c')
                x_sample = (x_sample * 255).astype(np.uint8)
                img = Image.fromarray(x_sample)
                img.save(os.path.join(output_path, f"image_{i:03d}.png"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_path', type=str, required=True)
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    args = parser.parse_args()

    config = OmegaConf.load("stable-diffusion/configs/stable-diffusion/v1-inference.yaml")
    model = load_model_from_config(config, "stable-diffusion/ldm/models/stable-diffusion-v1/model.ckpt")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    with open(args.json_path, 'r') as f:
        data = json.load(f)

    for source_idx, source_data in data.items():
        print(f"Processing source {source_idx}")

        token_name = source_data["token_name"]
        prompts = source_data["prompt"]

        # Train the embedding
        embedding_path = train_textual_inversion(args, source_idx, source_data, config, model)

        # For each prompt, generate images
        for prompt_idx, prompt in enumerate(prompts):
            output_path = os.path.join(args.output_dir, source_idx, str(prompt_idx))
            os.makedirs(output_path, exist_ok=True)

            generate_images(prompt, output_path, model, num_images=25)

            # Rename images according to the required format
            for i, img_file in enumerate(sorted(os.listdir(output_path))):
                if img_file.endswith('.png'):
                    old_path = os.path.join(output_path, img_file)
                    new_path = os.path.join(output_path, f"source{source_idx}_prompt{prompt_idx}_{i:02d}.png")
                    os.rename(old_path, new_path)


if __name__ == "__main__":
    main()
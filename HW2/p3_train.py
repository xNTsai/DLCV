# p3_train.py
import os
import json
import torch
import argparse
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from omegaconf import OmegaConf
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from ldm.util import instantiate_from_config
import torch.nn as nn

class MultiConceptTextualInversionDataset(Dataset):
    def __init__(self, data_root, data, size=512):
        self.size = size

        self.image_paths = []
        self.placeholder_tokens = []
        self.captions = []

        # Mapping from src_image to directory names
        src_image_to_dir = {
            'dog': '0',
            'David Revoy': '1',
        }

        # For each source in data
        for source_idx, source_data in data.items():
            src_image = source_data["src_image"]
            token_name = source_data["token_name"]
            source_dir_name = src_image_to_dir.get(src_image)
            if source_dir_name is None:
                raise ValueError(f"Unknown src_image '{src_image}' in input data.")
            source_dir = os.path.join(data_root, source_dir_name)
            image_paths = []
            for ext in ['*.jpg', '*.jpeg', '*.png']:
                image_paths.extend(list(Path(source_dir).glob(ext)))
            if not image_paths:
                raise ValueError(f"No images found in {source_dir}")
            self.image_paths.extend(image_paths)
            self.placeholder_tokens.extend([token_name] * len(image_paths))

            # Set appropriate caption template
            if src_image == 'dog':
                caption_template = f"a photo of {token_name}"
            elif src_image == 'David Revoy':
                caption_template = f"an artwork in the style of {token_name}"
            else:
                caption_template = f"a photo of {token_name}"

            self.captions.extend([caption_template] * len(image_paths))

        self.transform = transforms.Compose([
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    def __len__(self):
        return len(self.image_paths) * 100  # Adjust the repeats as needed

    def __getitem__(self, idx):
        idx = idx % len(self.image_paths)
        image = Image.open(self.image_paths[idx]).convert('RGB')
        image = self.transform(image)
        caption = self.captions[idx]
        return image, caption

def load_model_from_config(config, ckpt):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "state_dict" in pl_sd:
        sd = pl_sd["state_dict"]
    else:
        sd = pl_sd
    model = instantiate_from_config(config.model)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"Missing keys: {missing}")
    if unexpected:
        print(f"Unexpected keys: {unexpected}")
    model.cuda()
    model.eval()
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_path', type=str, required=True)
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--ckpt_dir', type=str, required=True)
    args = parser.parse_args()

    config = OmegaConf.load("stable-diffusion/configs/stable-diffusion/v1-inference.yaml")
    model = load_model_from_config(config, "stable-diffusion/ldm/models/stable-diffusion-v1/model.ckpt")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    tokenizer = model.cond_stage_model.tokenizer
    embeddings = model.cond_stage_model.transformer.get_input_embeddings()

    with open(args.json_path, 'r') as f:
        data = json.load(f)

    # Collect all placeholder tokens and their initializers
    placeholder_tokens = []
    initializer_tokens = {}
    for source_idx, source_data in data.items():
        placeholder_token = source_data["token_name"]
        placeholder_tokens.append(placeholder_token)
        src_image = source_data["src_image"]
        # Set appropriate initializer token
        if src_image == 'dog':
            initializer_token = 'dog'
        elif src_image == 'David Revoy':
            initializer_token = 'painting'
        else:
            initializer_token = 'art'  # Default initializer
        initializer_tokens[placeholder_token] = initializer_token

    # Add placeholder tokens to tokenizer and initialize embeddings
    token_id_to_embedding_param = {}

    for placeholder_token in placeholder_tokens:
        num_added_tokens = tokenizer.add_tokens([placeholder_token])
        if num_added_tokens == 0:
            print(f"Token {placeholder_token} already exists in tokenizer")
        token_id = tokenizer.convert_tokens_to_ids(placeholder_token)
        # Resize embeddings
        model.cond_stage_model.transformer.resize_token_embeddings(len(tokenizer))
        embeddings = model.cond_stage_model.transformer.get_input_embeddings()
        # Initialize embedding
        initializer_token = initializer_tokens[placeholder_token]
        initializer_token_id = tokenizer.convert_tokens_to_ids(initializer_token)
        with torch.no_grad():
            new_embedding = embeddings.weight[initializer_token_id].clone().detach()
        new_embedding = nn.Parameter(new_embedding)
        token_id_to_embedding_param[token_id] = new_embedding

    # Replace embeddings.forward
    embeddings_old_forward = embeddings.forward

    def embeddings_forward(input_ids, **kwargs):
        inputs_embeds = embeddings_old_forward(input_ids, **kwargs)
        for token_id, embedding_param in token_id_to_embedding_param.items():
            mask = input_ids == token_id
            if mask.any():
                inputs_embeds[mask] = embedding_param
        return inputs_embeds

    embeddings.forward = embeddings_forward

    # Set up optimizer
    optimizer = torch.optim.AdamW(list(token_id_to_embedding_param.values()), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=1e-6)

    # Prepare dataset
    dataset = MultiConceptTextualInversionDataset(args.input_dir, data)
    batch_size = 1
    accumulation_steps = 4
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.train()
    num_epochs = 10  # You can adjust the number of epochs

    best_loss = float('inf')
    best_epoch = 0

    for epoch in range(num_epochs):
        total_loss = 0
        optimizer.zero_grad()
        for i, batch in enumerate(tqdm(dataloader, desc=f"Training Epoch {epoch+1}/{num_epochs}")):
            images, captions = batch
            images = images.to(device)
            captions = list(captions)

            # Encode images to latents
            with torch.no_grad():
                latents = model.get_first_stage_encoding(model.encode_first_stage(images))

            # Get conditioning
            c = model.get_learned_conditioning(captions)

            # Sample random timesteps
            t = torch.randint(0, model.num_timesteps, (latents.shape[0],), device=device).long()

            # Add noise to the latents
            noise = torch.randn_like(latents)
            noisy_latents = model.q_sample(latents, t, noise=noise)

            # Predict the noise
            noise_pred = model.apply_model(noisy_latents, t, c)

            # Compute loss
            loss = F.mse_loss(noise_pred, noise) / accumulation_steps

            loss.backward()
            total_loss += loss.item() * accumulation_steps

            if (i + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

        scheduler.step()
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}, Average Loss: {avg_loss:.4f}")

        # Save the model if loss decreases
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_epoch = epoch + 1
            # Update embeddings.weight with the trained embeddings
            with torch.no_grad():
                for token_id, embedding_param in token_id_to_embedding_param.items():
                    embeddings.weight.data[token_id] = embedding_param.detach()
            # Save the best model
            os.makedirs(args.ckpt_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(args.ckpt_dir, "best_model.pt"))
            print(f"Best model saved to {os.path.join(args.ckpt_dir, 'best_model.pt')} at epoch {best_epoch}")

    # Save the last model
    os.makedirs(args.ckpt_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(args.ckpt_dir, "last_model.pt"))
    print(f"Last model saved to {os.path.join(args.ckpt_dir, 'last_model.pt')}")

    print(f"Training complete. Best model saved at epoch {best_epoch} with loss {best_loss:.4f}")

if __name__ == "__main__":
    main()
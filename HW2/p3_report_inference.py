# p3_inference.py
import os
import json
import torch
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config
from einops import rearrange
from ldm.models.diffusion.dpm_solver import DPMSolverSampler

def load_model_from_config(config, ckpt, placeholder_tokens):
    print(f"Loading model from {ckpt}")
    # Instantiate the model first
    model = instantiate_from_config(config.model)
    model.cuda()
    model.eval()
    
    # Access the tokenizer
    tokenizer = model.cond_stage_model.tokenizer
    embeddings = model.cond_stage_model.transformer.get_input_embeddings()
    
    # Add tokens to tokenizer and resize embeddings
    for token in placeholder_tokens:
        num_added_tokens = tokenizer.add_tokens([token])
        if num_added_tokens > 0:
            model.cond_stage_model.transformer.resize_token_embeddings(len(tokenizer))
            embeddings = model.cond_stage_model.transformer.get_input_embeddings()
    
    # Now load the state dict
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "state_dict" in pl_sd:
        sd = pl_sd["state_dict"]
    else:
        sd = pl_sd
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"Missing keys when loading state_dict: {missing}")
    if unexpected:
        print(f"Unexpected keys when loading state_dict: {unexpected}")
    return model

def generate_images(prompt, output_path, model, num_images=5, steps=50, batch_size=1, guidance_scale=7.5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(output_path, exist_ok=True)

    sampler = DPMSolverSampler(model)
    C = 4  # Latent channels
    H = W = 512  # Image size
    f = 8  # Downsampling factor

    with torch.no_grad():
        num_batches = (num_images + batch_size - 1) // batch_size
        image_counter = 0
        for _ in tqdm(range(num_batches), desc=f"Generating images for prompt '{prompt[:50]}...'"):
            current_batch_size = min(batch_size, num_images - image_counter)
            uc = model.get_learned_conditioning(current_batch_size * [""])
            c = model.get_learned_conditioning(current_batch_size * [prompt])
            shape = [C, H // f, W // f]
            samples, _ = sampler.sample(
                S=steps,
                conditioning=c,
                batch_size=current_batch_size,
                shape=shape,
                verbose=False,
                unconditional_guidance_scale=guidance_scale,
                unconditional_conditioning=uc,
                eta=0.0
            )
            x_samples = model.decode_first_stage(samples)
            x_samples = (x_samples + 1.0) / 2.0
            x_samples = torch.clamp(x_samples, 0.0, 1.0)

            # Save images
            for x_sample in x_samples:
                x_sample = x_sample.cpu().numpy()
                x_sample = rearrange(x_sample, 'c h w -> h w c')
                x_sample = (x_sample * 255).astype(np.uint8)
                img = Image.fromarray(x_sample)
                img.save(os.path.join(output_path, f"image_{image_counter:02d}.png"))
                image_counter += 1

def main():
    # Set random seed for reproducibility
    seed = 123
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    parser = argparse.ArgumentParser()
    parser.add_argument('ckpt_path', type=str)
    parser.add_argument('output_dir', type=str)
    args = parser.parse_args()

    # Define the placeholder tokens
    placeholder_tokens = ["<newdog>", "<newcat>", "<newartist>"]

    config = OmegaConf.load("stable-diffusion/configs/stable-diffusion/v1-inference.yaml")
    model = load_model_from_config(config, args.ckpt_path, placeholder_tokens)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Prompts with multiple concepts
    prompts = [
        "A <newdog> next to a cat in a park.",
        "A portrait of a <newdog> and a cat in the style of <newartist>.",
        "A cat sitting on top of a <newdog>."
    ]

    for idx, prompt in enumerate(prompts):
        output_path = os.path.join(args.output_dir, f"prompt_{idx}")
        os.makedirs(output_path, exist_ok=True)

        # Adjust guidance scale if needed
        if "<newartist>" in prompt:
            guidance_scale = 8.5
        else:
            guidance_scale = 7.5

        generate_images(
            prompt,
            output_path,
            model,
            num_images=10,
            steps=50,
            batch_size=1,
            guidance_scale=guidance_scale
        )

    print("Inference completed successfully.")

if __name__ == "__main__":
    main()

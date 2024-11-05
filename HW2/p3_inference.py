# p3_inference.py
import os
import sys
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

def generate_images(prompt, output_path, model, num_images=25, steps=50, batch_size=4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(output_path, exist_ok=True)

    sampler = DPMSolverSampler(model)
    C = 4  # Latent channels
    H = W = 512  # Image size
    f = 8  # Downsampling factor

    with torch.no_grad():
        num_batches = num_images // batch_size + int(num_images % batch_size > 0)
        image_counter = 0
        for _ in tqdm(range(num_batches), desc=f"Generating images for prompt '{prompt[:50]}...'"):
            current_batch_size = min(batch_size, num_images - image_counter)
            uc = model.get_learned_conditioning(current_batch_size * [""])
            c = model.get_learned_conditioning(current_batch_size * [prompt])
            shape = [C, H // f, W // f]
            samples, _ = sampler.sample(S=steps,
                                        conditioning=c,
                                        batch_size=current_batch_size,
                                        shape=shape,
                                        verbose=False,
                                        unconditional_guidance_scale=7.5,
                                        unconditional_conditioning=uc,
                                        eta=0.0)
            x_samples = model.decode_first_stage(samples)
            minVal = x_samples.min()
            maxVal = x_samples.max()
            if minVal < maxVal:
                x_samples = (x_samples - minVal) / (maxVal - minVal)
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
    parser.add_argument('json_path', type=str)
    parser.add_argument('output_dir', type=str)
    # parser.add_argument('ckpt_path', type=str)
    args = parser.parse_args()
    ckpt_path = 'hw2_p3_model_best.pt'
    
    with open(args.json_path, 'r') as f:
        data = json.load(f)

    # Collect placeholder tokens from the JSON file
    placeholder_tokens = []
    for source_idx, source_data in data.items():
        placeholder_tokens.append(source_data["token_name"])

    config = OmegaConf.load("stable-diffusion/configs/stable-diffusion/v1-inference.yaml")
    model = load_model_from_config(config, ckpt_path, placeholder_tokens)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    for source_idx, source_data in data.items():
        token_name = source_data["token_name"]
        prompts = source_data["prompt"]

        for prompt_idx, prompt in enumerate(prompts):
            output_path = os.path.join(args.output_dir, source_idx, str(prompt_idx))
            os.makedirs(output_path, exist_ok=True)

            generate_images(prompt, output_path, model, num_images=25, steps=50, batch_size=1)

            # Rename images according to the required format
            img_files = [f for f in sorted(os.listdir(output_path)) if f.endswith('.png')]
            for i, img_file in enumerate(img_files):
                old_path = os.path.join(output_path, img_file)
                new_path = os.path.join(output_path, f"source{source_idx}_prompt{prompt_idx}_{i:02d}.png")
                os.rename(old_path, new_path)

    print("Inference completed successfully.")

if __name__ == "__main__":
    main()


# ===============================================start evaluation================================================

# Image source: "dog", text prompt: A dog shepherd posing proudly on a hilltop with Mount Fuji in the background.
# CLIP Image Score: 81.08
# CLIP Text Score: 35.79

# =====================================================PASS======================================================

# Image source: "dog", text prompt: A dog perched on a park bench with the Colosseum looming behind.
# CLIP Image Score: 76.84
# CLIP Text Score: 35.19

# =====================================================PASS======================================================

# Image source: "David Revoy", text prompt: The streets of Paris in the style of David Revoy.
# CLIP Image Score: 70.43
# CLIP Text Score: 33.02

# =====================================================PASS======================================================

# Image source: "David Revoy", text prompt: Manhattan skyline in the style of David Revoy.
# CLIP Image Score: 70.12
# CLIP Text Score: 32.47

# =====================================================PASS======================================================

# You have passed 4/4 cases
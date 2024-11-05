import glob
import os
from copy import deepcopy

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class P2Dataset(Dataset):
    def __init__(self, path, transform, train=False):
        super().__init__()

        self.train = train
        self.transform = transform

        self.image_paths = sorted(glob.glob(os.path.join(path, "*.jpg")))

        if self.train:
            self.mask_paths = sorted(glob.glob(os.path.join(path, "*.png")))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image = Image.open(self.image_paths[index]).convert('RGB')  # Convert image to RGB
        image = self.transform(image)
        if self.train:
            mask = Image.open(self.mask_paths[index]).convert('RGB')  # Ensure mask is in RGB format
            mask = np.array(mask)
            mask = (mask >= 128).astype(int)
            # Combine the channels to create class indices
            mask = 4 * mask[:, :, 0] + 2 * mask[:, :, 1] + mask[:, :, 2]

            raw_mask = deepcopy(mask)

            mask[raw_mask == 3] = 0  # (Cyan: 011) Urban land
            mask[raw_mask == 6] = 1  # (Yellow: 110) Agriculture land
            mask[raw_mask == 5] = 2  # (Purple: 101) Rangeland
            mask[raw_mask == 2] = 3  # (Green: 010) Forest land
            mask[raw_mask == 1] = 4  # (Blue: 001) Water
            mask[raw_mask == 7] = 5  # (White: 111) Barren land
            mask[raw_mask == 0] = 6  # (Black: 000) Unknown

            mask = torch.tensor(mask, dtype=torch.long)
            # Ensure mask has shape [H, W]
            if mask.dim() != 2:
                mask = mask.squeeze()
            return image, mask
        else:
            return image, os.path.basename(self.image_paths[index])
import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from PIL import Image
import os

class miniDataset(Dataset):
    def __init__(self, folder_path, transform):
        self.Images = []
        self.Transformation = transform
        file_list = [file for file in os.listdir(folder_path) if file.endswith('.jpg')]
        for filename in file_list:
            # print(os.path.join(folder_path, filename))
            self.Images.append(Image.open(os.path.join(folder_path, filename)).convert('RGB'))

    def __getitem__(self, index):
        return self.Transformation(self.Images[index])

    def __len__(self):
        return len(self.Images)


if __name__ == "__main__":
    # calculates mean and std of training set
    train_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    dst = miniDataset("./hw1_data/p1_data/mini/train", train_transform)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    for x in dst:
        mean += x.mean(dim=(1, 2))
        std += x.std(dim=(1, 2))
    mean /= len(dst)
    std /= len(dst)
    print(mean, std)
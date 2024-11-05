import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from PIL import Image
import os

class officeDataset(Dataset):
    def __init__(self, folder_path, transform, train=True):
        self.Images = []
        self.Labels = []
        self.Transformation = transform
        self.Train = train
        self.num_classes = 65
        file_list = [file for file in os.listdir(folder_path) if file.endswith('.jpg')]
        for filename in file_list:
            try:
                image = Image.open(os.path.join(folder_path, filename)).convert('RGB')
                self.Images.append(image)
                if train:
                    label = int(filename.split("_")[0])
                    # print(f"Loaded label: {label}")
                    self.Labels.append(label)
            except Exception as e:
                print(f"Error loading file {filename}: {e}")

    def __getitem__(self, index):
        if self.Train:
            image = self.Transformation(self.Images[index])
            label = self.Labels[index]
            return image, label
        else:
            image = self.Transformation(self.Images[index])
            return image

    def __len__(self):
        return len(self.Images)


if __name__ == "__main__":
    # calculates mean and std of training set
    train_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    dst = officeDataset("./hw1_data/p1_data/office/train", train_transform)
    # dst = officeDataset("/kaggle/input/dlcv-hw1-data/hw1_data/p1_data/office/train", train_transform)

    mean = torch.zeros(3)
    std = torch.zeros(3)
    for x, _ in dst:
        print('success')
        mean += x.mean(dim=(1, 2))
        std += x.std(dim=(1, 2))
    mean /= len(dst)
    std /= len(dst)
    print(mean, std)
    print(f"Number of classes: {dst.num_classes}")
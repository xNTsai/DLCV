import os
import sys
import torch
import imageio
import numpy as np
from torchvision import transforms
from p2_model import DeepLabv3
from p2_dataloader import P2Dataset

def pred2image(batch_preds, batch_names, out_path):
    # batch_preds = (b, H, W)
    for pred, name in zip(batch_preds, batch_names):
        pred = pred.detach().cpu().numpy()
        pred_img = np.zeros((512, 512, 3), dtype=np.uint8)
        pred_img[np.where(pred == 0)] = [0, 255, 255]  # (Cyan: 011) Urban land 
        pred_img[np.where(pred == 1)] = [255, 255, 0]  # (Yellow: 110) Agriculture land
        pred_img[np.where(pred == 2)] = [255, 0, 255]  # (Purple: 101) Rangeland
        pred_img[np.where(pred == 3)] = [0, 255, 0]    # (Green: 010) Forest land
        pred_img[np.where(pred == 4)] = [0, 0, 255]    # (Blue: 001) Water
        pred_img[np.where(pred == 5)] = [255, 255, 255]  # (White: 111) Barren land
        pred_img[np.where(pred == 6)] = [0, 0, 0]      # (Black: 000) Unknown
        imageio.imwrite(os.path.join(out_path, name.replace('_sat.jpg', '_mask.png')), pred_img)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    input_folder = sys.argv[1]
    output_folder = sys.argv[2]

    os.makedirs(output_folder, exist_ok=True)

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    test_dataset = P2Dataset(input_folder, transform=test_transform, train=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=4)

    net = DeepLabv3(n_classes=7, mode='resnet')
    net.load_state_dict(torch.load('./hw1_p2_best_model.pth', map_location=device, weights_only=True))
    # net.load_state_dict(torch.load('./P2_B_checkpoint/best_model.pth', map_location=device, weights_only=True))
    net = net.to(device)
    net.eval()

    with torch.no_grad():
        for images, filenames in test_loader:
            images = images.to(device)
            outputs = net(images)
            preds = outputs['out'].argmax(dim=1)
            pred2image(preds, filenames, output_folder)

    print("Inference completed. Results saved in", output_folder)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python inference.py <input_folder> <output_folder>")
        sys.exit(1)
    main()
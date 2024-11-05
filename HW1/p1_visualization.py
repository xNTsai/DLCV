import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from tqdm import tqdm

def visualize_tSNE(model, dataloader, device, epoch, plot_path, model_name, train=True):
    model.eval()
    all_x = None
    all_y = None
    
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc=f'Extracting features for epoch {epoch}'):
            inputs = inputs.to(device)
            features = model.get_embedding(inputs)
            if all_x is None:
                all_x = features.detach().cpu().numpy()
                all_y = labels.numpy().flatten()
            else:
                features = features.detach().cpu().numpy()
                all_x = np.vstack((all_x, features))
                all_y = np.concatenate((all_y, labels.numpy().flatten()))
    
    all_x = all_x.reshape(all_x.shape[0], -1)
    
    # t-SNE visualization
    tsne = TSNE(n_components=2, random_state=42)
    tsne_result = tsne.fit_transform(all_x)
    plt.figure(figsize=(10, 8))
    plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=all_y, cmap='viridis')
    plt.colorbar()
    plt.title(f"t-SNE visualization for epoch {epoch}")
    if train:
        plt.savefig(f"{plot_path}/{model_name}_TSNE_epoch_{epoch}.png")
    else:
        plt.savefig(f"{plot_path}/{model_name}_TSNE_epoch_{epoch}_test.png")
    plt.close()
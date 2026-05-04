# plot_error_distribution.py
import os
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import r2_score
from experiments.experiment_6 import EyeGazeDataset, get_transform, EyeGazeNetV3
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def calculate_metrics(model, dataloader, device):
    model.eval()
    x_errors, y_errors, euclidean_errors = [], [], []

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            preds = model(images)
            diff = labels - preds
            x_errors.append(diff[:, 0].cpu())
            y_errors.append(diff[:, 1].cpu())
            euclidean_errors.append(torch.norm(diff, dim=1).cpu())

    x_errors = torch.cat(x_errors)
    y_errors = torch.cat(y_errors)
    euclidean_errors = torch.cat(euclidean_errors)

    return {
        'MAE_X': torch.mean(torch.abs(x_errors)).item(),
        'MAE_Y': torch.mean(torch.abs(y_errors)).item(),
        'RMSE_X': torch.sqrt(torch.mean(x_errors ** 2)).item(),
        'RMSE_Y': torch.sqrt(torch.mean(y_errors ** 2)).item(),
        'Std_X': torch.std(x_errors).item(),
        'Std_Y': torch.std(y_errors).item(),
        'Euclidean_Error': torch.mean(euclidean_errors).item(),
        'Euclidean_Std': torch.std(euclidean_errors).item()
    }


def calculate_advanced_metrics(model, dataloader, device):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            preds = model(images)
            all_preds.append(preds.cpu().detach())
            all_labels.append(labels.cpu().detach())

    preds = torch.cat(all_preds)
    labels = torch.cat(all_labels)

    return {
        'R2_X': r2_score(labels[:, 0].numpy(), preds[:, 0].numpy()),
        'R2_Y': r2_score(labels[:, 1].numpy(), preds[:, 1].numpy()),
        'Median_Euclidean_Error': torch.median(torch.norm(labels - preds, dim=1)).item()
    }


def plot_error_distribution(model, dataloader, device):
    model.eval()
    x_errors, y_errors = [], []

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            preds = model(images)
            diff = preds - labels
            x_errors.extend(diff[:, 0].cpu().numpy())
            y_errors.extend(diff[:, 1].cpu().numpy())

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.hist(x_errors, bins=30, color='orange', alpha=0.7)
    plt.title("X Error Distribution")
    plt.axvline(np.mean(x_errors), color='red', linestyle='--', label=f'Mean: {np.mean(x_errors):.2f}')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.hist(y_errors, bins=30, color='skyblue', alpha=0.7)
    plt.title("Y Error Distribution")
    plt.axvline(np.mean(y_errors), color='red', linestyle='--', label=f'Mean: {np.mean(y_errors):.2f}')
    plt.legend()
    plt.show()


def plot_joint_error_distribution(model, dataloader, device):
    model.eval()
    x_errors, y_errors = [], []

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            preds = model(images)
            diff = preds - labels
            x_errors.extend(diff[:, 0].cpu().numpy())
            y_errors.extend(diff[:, 1].cpu().numpy())

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.scatter(x_errors, y_errors, alpha=0.5)
    plt.xlabel('X Error (pixels)')
    plt.ylabel('Y Error (pixels)')
    plt.title('Joint Error Distribution')
    plt.axhline(0, color='black', linestyle='--')
    plt.axvline(0, color='black', linestyle='--')

    plt.subplot(1, 2, 2)
    euclidean_errors = np.sqrt(np.square(x_errors) + np.square(y_errors))
    plt.hist(euclidean_errors, bins=30, color='purple', alpha=0.7)
    plt.xlabel('Euclidean Error (pixels)')
    plt.title('Euclidean Error Distribution')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    dataset_path = os.environ.get("DATASET_PATH", "./dataset")
    model_path = os.environ.get("GAZE_MODEL_PATH")
    if not model_path:
        raise EnvironmentError("Задай GAZE_MODEL_PATH — путь к .pt файлу модели")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    full_dataset = EyeGazeDataset(dataset_path, transform=get_transform())
    val_size = int(0.2 * len(full_dataset))
    _, val_set = random_split(full_dataset, [len(full_dataset) - val_size, val_size])
    val_set.dataset.transform = get_transform()
    val_loader = DataLoader(val_set, batch_size=32)

    model = torch.jit.load(model_path, map_location=device)

    print("\n=== Basic Metrics ===")
    for name, value in calculate_metrics(model, val_loader, device).items():
        print(f"{name}: {value:.2f} px")

    print("\n=== Advanced Metrics ===")
    for name, value in calculate_advanced_metrics(model, val_loader, device).items():
        print(f"{name}: {value:.4f}")

    plot_error_distribution(model, val_loader, device)
    plot_joint_error_distribution(model, val_loader, device)

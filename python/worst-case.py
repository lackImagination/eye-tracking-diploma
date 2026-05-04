# worst-case.py
import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import DataLoader
import random
from screeninfo import get_monitors
from torch.utils.data import Dataset, DataLoader, random_split
from main import EyeGazeDataset, EyeGazeResNetPix, get_transform, PadToSquare, get_tta_transforms, custom_collate_fn, tta_predict
from torchvision import transforms
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def denormalize(tensor):
    """Конвертирует тензор обратно в изображение для отображения"""
    return tensor.cpu().permute(1, 2, 0).numpy()


def visualize_worst_cases(model, dataloader, device, n=5, use_tta=True):
    model.eval()
    worst_cases = []

    with torch.no_grad():
        for images, labels, originals in dataloader:
            labels = labels.to(device)

            for i in range(len(images)):
                if use_tta and originals[i] is not None:
                    # Используем TTA если есть оригинальное изображение
                    tta_transforms = get_tta_transforms((224, 224))
                    pred = tta_predict(model, originals[i], tta_transforms, device)
                else:
                    # Обычный предсказание
                    pred = model(images[i].unsqueeze(0).to(device))

                error = torch.norm(pred - labels[i].unsqueeze(0)).item()

                worst_cases.append({
                    'image': originals[i] if originals[i] is not None else images[i].cpu(),
                    'actual': labels[i].cpu().numpy(),
                    'predicted': pred.squeeze(0).cpu().numpy(),
                    'error': error,
                    'is_original': originals[i] is not None
                })

    # Сортировка по ошибке
    worst_cases.sort(key=lambda x: x['error'], reverse=True)
    worst_cases = worst_cases[:n]

    # Визуализация
    for idx, case in enumerate(worst_cases):
        if case['is_original']:
            img = case['image']
        else:
            img = denormalize(case['image'])

        actual = case['actual']
        predicted = case['predicted']
        error = case['error']

        plt.figure(figsize=(8, 4))

        # Subplot 1: Изображение с точками
        plt.subplot(1, 2, 1)
        plt.imshow(img)
        plt.scatter(*actual, c='green', label='Actual', s=50)
        plt.scatter(*predicted, c='red', label='Predicted', s=50)
        plt.title(f"Error: {error:.2f}px (Top {idx + 1})")
        plt.legend()
        plt.axis('off')

        # Subplot 2: Координаты на экране
        plt.subplot(1, 2, 2)
        plt.xlim(0, screen_w)
        plt.ylim(screen_h, 0)  # Инверсия Y
        plt.scatter(*actual, c='green', label='Actual', s=50)
        plt.scatter(*predicted, c='red', label='Predicted', s=50)
        plt.title("Screen Coordinates")
        plt.legend()
        plt.grid(True)
        plt.xlabel("X")
        plt.ylabel("Y")

        plt.tight_layout()
        plt.show()


def plot_error_distribution(model, dataloader, device, use_tta=True):
    model.eval()
    x_errors, y_errors = [], []

    with torch.no_grad():
        for images, labels, originals in dataloader:
            labels = labels.to(device)

            for i in range(len(images)):
                if use_tta and originals[i] is not None:
                    tta_transforms = get_tta_transforms((224, 224))
                    pred = tta_predict(model, originals[i], tta_transforms, device)
                else:
                    pred = model(images[i].unsqueeze(0).to(device))

                diff = pred - labels[i].unsqueeze(0)
                x_errors.append(diff[0, 0].cpu().item())
                y_errors.append(diff[0, 1].cpu().item())

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


if __name__ == "__main__":
    # === Настройки ===
    dataset_path = os.environ.get("DATASET_PATH", "./dataset")
    model_path = "model/model_with_tta.pth"

    # === Определение экрана ===
    screen_info = get_monitors()[0]
    screen_w, screen_h = screen_info.width, screen_info.height

    # === Устройство ===
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # === Загрузка модели ===
    model = EyeGazeResNetPix().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    # === Датасет ===
    full_dataset = EyeGazeDataset(dataset_path, transform=get_transform(), return_original=True)

    # Разделение на валидацию
    val_size = int(0.2 * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_set, val_set = random_split(full_dataset, [train_size, val_size])

    # Трансформации для валидации
    val_set.dataset.transform = get_transform(augmented=False)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, collate_fn=custom_collate_fn)

    # === Визуализация ===
    print("\n==== Анализ worst-case ====")
    # visualize_worst_cases(model, val_loader, device, n=5, use_tta=True)
    plot_error_distribution(model, val_loader, device, use_tta=True)

    '''
    # Демонстрация PadToSquare
    img = Image.open("IMAGES_PATH", ".image").convert("RGB")
    square_img = PadToSquare()(img)
    square_img.show()
    '''
import os
import csv
import pandas as pd
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import torchvision.transforms.functional as TF
import torchvision.models as models
from torchvision.models import resnet18, ResNet18_Weights
from screeninfo import get_monitors
from torchsummary import summary
import ssl
import certifi
import random
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from datetime import datetime


# ====================== КОНФИГ ======================
class Config:
    def __init__(self):
        self.seed = 42
        self.batch_size = 32
        self.learning_rate = 1e-3
        self.epochs = 100
        self.patience = 5
        self.input_size = 224
        self.screen_width = 1440  # Ваши реальные размеры экрана
        self.screen_height = 900
        self.data_path = "/dataset"  # Укажите свой путь


# ====================== УТИЛИТЫ ======================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class PadToSquare:
    def __call__(self, image):
        w, h = image.size
        max_side = max(w, h)
        padding = (
            (max_side - w) // 2,
            (max_side - h) // 2,
            (max_side - w + 1) // 2,
            (max_side - h + 1) // 2
        )
        return transforms.functional.pad(image, padding, fill=0)


def get_transform(size=224):
    return transforms.Compose([
        PadToSquare(),
        transforms.Resize((size, size)),
        transforms.ToTensor(),
    ])


# ====================== ДАТАСЕТ ======================
class EyeGazeDataset(Dataset):
    def __init__(self, dataset_dir, transform=None):
        self.images_dir = os.path.join(dataset_dir, "images")
        self.annotations = pd.read_csv(
            os.path.join(dataset_dir, "coords.csv"),
            header=None,
            names=["image_name", "x", "y"]
        )
        self.transform = transform or get_transform()

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        row = self.annotations.iloc[idx]
        img = Image.open(os.path.join(self.images_dir, row['image_name'])).convert("RGB")

        if self.transform:
            img = self.transform(img)

        # Возвращаем координаты в пикселях без нормализации!
        coords = torch.tensor([row['x'], row['y']], dtype=torch.float32)
        return img, coords


# ====================== МОДЕЛЬ ======================
class EnhancedEyeGazeNet(nn.Module):
    def __init__(self, screen_width=1440, screen_height=900):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2, padding=1),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.regressor = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 2)
        )

        # инициализация последнего слоя
        nn.init.normal_(self.regressor[-1].weight, mean=0, std=0.001)
        nn.init.constant_(self.regressor[-1].bias, 500)  # Среднее значение координат

    def forward(self, x):
        x = self.features(x)
        return self.regressor(x)


# ====================== ТРЕНИРОВКА ======================
def train_model(config):
    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Создаем папку для сохранения результатов
    os.makedirs("training_results_4", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"training_results_4/{timestamp}"
    os.makedirs(results_dir, exist_ok=True)

    # Инициализируем CSV файл для логирования
    log_file = os.path.join(results_dir, "training_log.csv")
    with open(log_file, 'w') as f:
        f.write("epoch,train_loss,val_loss,pixel_error\n")

    # Данные
    dataset = EyeGazeDataset(config.data_path)
    train_size = int(0.8 * len(dataset))
    train_set, val_set = random_split(dataset, [train_size, len(dataset) - train_size])

    train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=config.batch_size)

    # Модель и оптимизатор
    model = EnhancedEyeGazeNet().to(device)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=3e-4,
        weight_decay=1e-6,
        betas=(0.9, 0.999)
    )
    criterion = nn.L1Loss()

    # Для хранения истории обучения
    history = {'epoch': [], 'train_loss': [], 'val_loss': [], 'pixel_error': []}

    best_loss = float('inf')
    patience_counter = 0

    for epoch in range(config.epochs):
        # Тренировка
        model.train()
        train_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Валидация
        model.eval()
        val_loss = 0.0
        pixel_error = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)

                val_loss += criterion(outputs, labels).item()
                pixel_error += torch.mean(torch.abs(outputs - labels)).item()

        # Нормализация метрик
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        pixel_error /= len(val_loader)

        # Сохраняем метрики
        history['epoch'].append(epoch + 1)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['pixel_error'].append(pixel_error)

        # Запись в CSV
        with open(log_file, 'a') as f:
            f.write(f"{epoch + 1},{train_loss:.6f},{val_loss:.6f},{pixel_error:.6f}\n")

        print(f"Epoch {epoch + 1}/{config.epochs} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Pixel Error: {pixel_error:.2f}px")

        # Early stopping и сохранение модели
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            traced_model = torch.jit.script(model)
            traced_model.save(f"{results_dir}/model_4.pt")
            print(f"Model saved with val_loss: {val_loss:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                print("Early stopping!")
                break

    # Визуализация кривых обучения
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history['epoch'], history['train_loss'], label='Train Loss')
    plt.plot(history['epoch'], history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history['epoch'], history['pixel_error'], color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Pixels')
    plt.title('Pixel Error')

    plt.tight_layout()
    plt.savefig(f"{results_dir}/training_plots.png")
    plt.close()

    # Сохраняем конфиг
    with open(f"{results_dir}/config.txt", 'w') as f:
        for key, value in vars(config).items():
            f.write(f"{key}: {value}\n")

    print(f"\nTraining results saved to: {results_dir}")
    print(f"Best model: {results_dir}/best_model.pt")
    print(f"Training log: {results_dir}/training_log.csv")
    print(f"Training plots: {results_dir}/training_plots.png")


def print_model_summary(model, input_size=(3, 224, 224)):
    summary(model.to('cuda' if torch.cuda.is_available() else 'cpu'), input_size)


# ====================== ЗАПУСК ======================
if __name__ == "__main__":
    config = Config()
    print_model_summary(EnhancedEyeGazeNet())
    print("Starting training...")
    train_model(config)
    print("Training completed! Model saved as 'eyegaze_model.pt'")
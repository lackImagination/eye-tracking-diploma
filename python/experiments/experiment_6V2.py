import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import torch.nn.functional as F
from PIL import Image
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import random


# ====================== КОНФИГ ======================
class Config:
    def __init__(self):
        self.seed = 42
        self.batch_size = 32
        self.learning_rate = 3e-4
        self.epochs = 100
        self.patience = 20
        self.input_size = 224
        self.screen_width = 1440
        self.screen_height = 900
        self.data_path = "/dataset"
        self.save_dir = "training_results_finetune"


# ====================== УТИЛИТЫ ======================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_save_dir(base_dir):
    """Создает уникальную директорию для сохранения результатов"""
    os.makedirs(base_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(base_dir, f"run_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)
    return save_dir


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


def get_transform(train=True):
    """Упрощенные трансформации для fine-tuning"""
    base_transform = [
        PadToSquare(),
        transforms.Resize((224, 224)),
    ]

    if train:
        # Только мягкие аугментации
        base_transform.extend([
            transforms.RandomApply([transforms.ColorJitter(brightness=0.1, contrast=0.1)], p=0.3),
        ])

    base_transform.append(transforms.ToTensor())
    return transforms.Compose(base_transform)


# ====================== ДАТАСЕТ ======================
class EyeGazeDataset(Dataset):
    def __init__(self, dataset_dir, train=True):
        self.images_dir = os.path.join(dataset_dir, "images")
        self.annotations = pd.read_csv(
            os.path.join(dataset_dir, "coords.csv"),
            header=None,
            names=["image_name", "x", "y"]
        )
        self.transform = get_transform(train)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        row = self.annotations.iloc[idx]
        img = Image.open(os.path.join(self.images_dir, row['image_name'])).convert("RGB")

        if self.transform:
            img = self.transform(img)

        coords = torch.tensor([row['x'], row['y']], dtype=torch.float32)
        return img, coords


# ====================== МОДЕЛЬ ======================
class EyeGazeNetV3(nn.Module):
    def __init__(self):
        super().__init__()

        # Initial block
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2, padding=1)
        )

        # Residual blocks
        self.block2 = nn.ModuleDict({
            'main': nn.Sequential(
                nn.Conv2d(64, 64, 3, stride=1, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(64, 64, 3, padding=1),
                nn.BatchNorm2d(64)
            ),
            'shortcut': nn.Identity()
        })

        self.block3 = self._make_res_block(64, 128, downsample=True)
        self.block4 = self._make_res_block(128, 256, downsample=True)
        self.block5 = self._make_res_block(256, 512, downsample=True)

        # Attention
        self.attention = nn.Sequential(
            nn.Conv2d(512, 512, 1),
            nn.ReLU(),
            nn.Conv2d(512, 512, 1),
            nn.Sigmoid()
        )

        # Regressor
        self.regressor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.Dropout(0.3),
            nn.Linear(256, 2)
        )

        # Инициализация
        nn.init.normal_(self.regressor[-1].weight, mean=0, std=0.001)
        nn.init.constant_(self.regressor[-1].bias, 500)

    def _make_res_block(self, in_channels, out_channels, downsample=False):
        layers = []
        stride = 2 if downsample else 1

        # Main branch
        layers.extend([
            nn.Conv2d(in_channels, out_channels, 3, stride, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels)
        ])

        # Shortcut connection
        shortcut = nn.Sequential()
        if in_channels != out_channels or downsample:
            shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride),
                nn.BatchNorm2d(out_channels)
            )

        return nn.ModuleDict({'main': nn.Sequential(*layers), 'shortcut': shortcut})

    def forward(self, x):
        x = self.conv1(x)

        # Residual blocks
        for block in [self.block2, self.block3, self.block4, self.block5]:
            residual = block['shortcut'](x)
            x = block['main'](x)
            x = F.relu(x + residual)

        # Attention
        att = self.attention(x)
        x = x * att

        return self.regressor(x)


def unfreeze_layers(model):
    """Размораживаем больше слоёв для лучшего обучения"""
    # Полностью размораживаем последние 3 блока + attention + regressor
    for name, param in model.named_parameters():
        if any(key in name for key in ['block3', 'block4', 'block5', 'attention', 'regressor']):
            param.requires_grad = True
            print(f"Разморожен: {name}")
        else:
            param.requires_grad = False


def get_optimizer(model):
    # Разные learning rates для разных слоёв
    params = [
        {'params': [p for n,p in model.named_parameters() if 'block' in n], 'lr': 3e-4},
        {'params': [p for n,p in model.named_parameters() if 'attention' in n], 'lr': 1e-4},
        {'params': [p for n,p in model.named_parameters() if 'regressor' in n], 'lr': 5e-4}
    ]
    return optim.AdamW(params, weight_decay=1e-5)  # Добавили регуляризацию


# Улучшенный learning rate scheduler:
def get_scheduler(optimizer):
    return torch.optim.lr_scheduler.CyclicLR(
        optimizer,
        base_lr=1e-5,
        max_lr=5e-5,
        step_size_up=5,
        cycle_momentum=False
    )


# ====================== ТРЕНИРОВКА ======================
def save_training_plots(history, save_dir):
    """Сохраняет графики обучения"""
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
    plt.savefig(os.path.join(save_dir, "training_plots.png"))
    plt.close()


def train_model(config, model):
    set_seed(config.seed)
    device = next(model.parameters()).device
    save_dir = create_save_dir(config.save_dir)

    log_file = os.path.join(save_dir, "training_log.csv")
    with open(log_file, 'w') as f:
        f.write("epoch,train_loss,val_loss,pixel_error\n")

    dataset = EyeGazeDataset(config.data_path)
    train_size = int(0.8 * len(dataset))
    train_set, val_set = random_split(dataset, [train_size, len(dataset) - train_size])

    train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=config.batch_size)

    optimizer = get_optimizer(model)
    scheduler = get_scheduler(optimizer)
    criterion = nn.SmoothL1Loss()

    history = {
        'epoch': [],
        'train_loss': [],
        'val_loss': [],
        'pixel_error': []
    }

    best_loss = float('inf')
    patience_counter = 0

    for epoch in range(config.epochs):
        model.train()
        train_loss = 0.0

        if epoch < 5:
            for g in optimizer.param_groups:
                g['lr'] = min(g['lr'] * 1.2, 5e-5)

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            for name, param in model.named_parameters():
                if 'regressor' in name and param.grad is not None:
                    param.grad *= 2.0
            optimizer.step()

            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        pixel_error = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                val_loss += criterion(outputs, labels).item()
                pixel_error += torch.mean(torch.abs(outputs - labels)).item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        pixel_error /= len(val_loader)

        history['epoch'].append(epoch + 1)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['pixel_error'].append(pixel_error)

        with open(log_file, 'a') as f:
            f.write(f"{epoch + 1},{train_loss:.6f},{val_loss:.6f},{pixel_error:.6f}\n")

        print(f"Epoch {epoch + 1}/{config.epochs} | "
              f"Train Loss: {train_loss:.2f} | "
              f"Val Loss: {val_loss:.2f} | "
              f"Pixel Error: {pixel_error:.2f}px")

        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0

            # Сохраняем модель
            model_path = os.path.join(save_dir, "best_model.pt")
            torch.jit.script(model).save(model_path)

            save_training_plots(history, save_dir)
            print(f"Model saved with val_loss: {val_loss:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                print("Early stopping!")
                break

        scheduler.step(val_loss)

    save_training_plots(history, save_dir)

    with open(os.path.join(save_dir, "config.txt"), 'w') as f:
        for key, value in vars(config).items():
            f.write(f"{key}: {value}\n")

    print(f"\nTraining results saved to: {save_dir}")


# ====================== ЗАПУСК ======================
if __name__ == "__main__":
    config = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Загрузка модели
    pretrained_path = "/experiments/training_results_6V2/20250512_124019/model_6v2.pt"
    model = torch.jit.load(pretrained_path, map_location=device)
    model.train()

    # 3. Проверка параметров (исправленная версия)
    print("\nОбучаемые параметры:\n\n")
    total_params = 0
    trainable_params = 0

    # Для TorchScript моделей используем state_dict() для доступа к параметрам
    for name, param in model.state_dict().items():
        param_tensor = model.state_dict()[name]
        total_params += param_tensor.numel()
        # Проверяем, есть ли этот параметр в requires_grad
        if name in [n for n, p in model.named_parameters()]:
            trainable_params += param_tensor.numel()

    # 4. Настройка fine-tuning для TorchScript
    for name, param in model.named_parameters():
        param.requires_grad = True

        # Запуск обучения
    train_model(config, model)
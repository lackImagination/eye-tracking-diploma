# data.py
import os
import pandas as pd
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
import ssl
import certifi
import numpy as np
import random
import matplotlib
matplotlib.use('TkAgg')
ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())


# === Стабильность ===
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# === Утилиты ===
class PadToSquare:
    def __call__(self, image):
        w, h = image.size
        max_side = max(w, h)
        padding = (
            (max_side - w) // 2,  # left
            (max_side - h) // 2,  # top
            (max_side - w + 1) // 2,  # right
            (max_side - h + 1) // 2  # bottom
        )
        return TF.pad(image, padding, fill=0, padding_mode='constant')


def get_transform(target_size=224):
    return transforms.Compose([
        PadToSquare(),
        transforms.Resize((target_size, target_size)),
        transforms.ToTensor()
    ])


# === Dataset ===
class EyeGazeDataset(Dataset):
    def __init__(self, dataset_dir, transform=None, screen_width=1920, screen_height=1080):
        self.dataset_dir = dataset_dir
        self.images_dir = os.path.join(dataset_dir, "images")
        self.annotations = pd.read_csv(
            os.path.join(dataset_dir, "coords.csv"),
            sep=",",
            header=None,
            names=["image_name", "x", "y"]
        )

        self.screen_width = screen_width
        self.screen_height = screen_height
        self.transform = transform if transform else get_transform()

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        row = self.annotations.iloc[idx]
        image_path = os.path.join(self.images_dir, row['image_name'])
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        x = torch.tensor(row['x'] / self.screen_width, dtype=torch.float32)
        y = torch.tensor(row['y'] / self.screen_height, dtype=torch.float32)
        label = torch.tensor([x, y], dtype=torch.float32)

        return image, label


# === Модель_1 ===
class EyeGazeCNN(nn.Module):
    def __init__(self):
        super(EyeGazeCNN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )

        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 28 * 28, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 2)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.regressor(x)
        return x


class EyeGazeResNet(nn.Module):
    def __init__(self, pretrained=True):
        super(EyeGazeResNet, self).__init__()

        # Загружаем предобученную ResNet18
        self.base_model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        self.base_model.fc = nn.Sequential(
            nn.Linear(self.base_model.fc.in_features, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.base_model(x)


# === Тренировка ===
def train_model(model, train_loader, val_loader, criterion, optimizer, device, screen_w, screen_h,
                num_epochs=30, patience=3):

    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_path = "model/best_model2.pth"

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            # print("outputs.shape: ", outputs.shape)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Валидация
        model.eval()
        val_loss = 0.0
        total_pixel_error = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                preds_px = outputs * torch.tensor([screen_w, screen_h], device=device)
                labels_px = labels * torch.tensor([screen_w, screen_h], device=device)
                pixel_error = torch.norm(preds_px - labels_px, dim=1).mean()
                total_pixel_error += pixel_error.item()

        val_loss /= len(val_loader)
        avg_pixel_error = total_pixel_error / len(val_loader)

        print(f"Epoch {epoch + 1}/{num_epochs} | Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | Pixel Error: {avg_pixel_error:.2f}px")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered at epoch {epoch + 1}")
                break

    print("Best model saved as:", best_model_path)


# === Прогнозирование ===
def predict_coords(model, image_tensor, screen_width, screen_height):
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.unsqueeze(0).to(device)
        output = model(image_tensor)
        coords = output[0] * torch.tensor([screen_width, screen_height], device=device)
        return coords.cpu().numpy()


# === Запуск ===
if __name__ == "__main__":
    set_seed(42)
    dataset_path = os.environ.get("DATASET_PATH", "./dataset")
    screen_info = get_monitors()[0]
    screen_w, screen_h = screen_info.width, screen_info.height
    print("screen_w, screen_h =", screen_w, screen_h)

    transform = get_transform()
    full_dataset = EyeGazeDataset(dataset_path, transform=transform, screen_width=screen_w, screen_height=screen_h)

    # Разделение данных (80% train, 20% val)
    val_ratio = 0.2
    val_size = int(len(full_dataset) * val_ratio)
    train_size = len(full_dataset) - val_size
    train_set, val_set = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = EyeGazeCNN().to(device)
    model = EyeGazeResNet(pretrained=True).to(device)

    criterion = nn.SmoothL1Loss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    train_model(model, train_loader, val_loader, criterion, optimizer, device, screen_w, screen_h)

    # Пример предсказания
    model.load_state_dict(torch.load("model/best_model2.pth"))
    model.eval()
    for images, labels in val_loader:
        images = images.to(device)
        preds = model(images) * torch.tensor([screen_w, screen_h], device=device)
        actuals = labels * torch.tensor([screen_w, screen_h], device=device)
        print("Predicted (px):", preds[:5].cpu())
        print("Actual (px):", actuals[:5])
        break

# main.py
import os
import ssl
import random
import certifi
import pandas as pd
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
from torchvision.models import resnet18, ResNet18_Weights
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchvision.transforms.functional as TF

ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())


# ==== Стабильность ====
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ==== Трансформации ====
class PadToSquare:
    def __call__(self, image):
        w, h = image.size
        max_side = max(w, h)
        padding = (
            (max_side - w) // 2,
            (max_side - h) // 2,
            (max_side - w + 1) // 2,
            (max_side - h + 1) // 2,
        )
        return TF.pad(image, padding, fill=0)


def get_transform(augmented=False, target_size=224):
    transform_list = [PadToSquare(), transforms.Resize((target_size, target_size))]
    if augmented:
        transform_list += [
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.RandomAffine(degrees=2, translate=(0.02, 0.02), scale=(0.95, 1.05)),
            transforms.GaussianBlur(kernel_size=3)
        ]
    transform_list.append(transforms.ToTensor())
    return transforms.Compose(transform_list)


def get_tta_transforms(image_size):
    return [
        transforms.Compose([PadToSquare(), transforms.Resize(image_size), transforms.ToTensor()]),
        transforms.Compose([PadToSquare(), transforms.Resize(image_size), transforms.ColorJitter(brightness=0.1), transforms.ToTensor()]),
        transforms.Compose([PadToSquare(), transforms.Resize(image_size), transforms.ColorJitter(contrast=0.1), transforms.ToTensor()]),
        transforms.Compose([PadToSquare(), transforms.RandomResizedCrop(image_size, scale=(0.95, 1.0)), transforms.ToTensor()]),
    ]


# ==== Dataset ====
class EyeGazeDataset(Dataset):
    def __init__(self, dataset_dir, transform=None, return_original=False):
        self.dataset_dir = dataset_dir
        self.images_dir = os.path.join(dataset_dir, "images")
        self.annotations = pd.read_csv(os.path.join(dataset_dir, "coords.csv"), header=None, names=["image_name", "x", "y"])
        self.transform = transform or get_transform()
        self.return_original = return_original

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        row = self.annotations.iloc[idx]
        image_path = os.path.join(self.images_dir, row["image_name"])
        image = Image.open(image_path).convert("RGB")
        transformed = self.transform(image)
        label = torch.tensor([row["x"], row["y"]], dtype=torch.float32)
        return transformed, label, image if self.return_original else None


# ==== Custom collate_fn ====
def custom_collate_fn(batch):
    images, labels, originals = zip(*batch)
    images = torch.stack(images)
    labels = torch.stack(labels)
    return images, labels, list(originals)


# ==== Модель ====
class EyeGazeResNetPix(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_model = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.base_model.fc = nn.Sequential(
            nn.Linear(self.base_model.fc.in_features, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        return self.base_model(x)


# ==== Метрика ====
def compute_pixel_error(preds, targets):
    return torch.norm(preds - targets, dim=1).mean().item()


# ==== TTA ====
def tta_predict(model, image, tta_transforms, device):
    model.eval()
    with torch.no_grad():
        preds = []
        for t in tta_transforms:
            img = t(image).unsqueeze(0).to(device)
            pred = model(img)
            preds.append(pred.cpu())
        return torch.stack(preds).mean(dim=0)


# ==== Обучение ====
def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=30, patience=5):
    best_loss = float('inf')
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=7, factor=0.5)
    early_stop_counter = 0

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        for images, labels, _ in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            preds = model(images)
            loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        val_loss = 0.0
        total_pixel_error = 0.0
        with torch.no_grad():
            for images, labels, originals in val_loader:
                labels = labels.to(device)
                for i in range(len(images)):
                    if originals[i] is not None:
                        pred = tta_predict(model, originals[i], get_tta_transforms((224, 224)), device)
                    else:
                        pred = model(images[i].unsqueeze(0).to(device))
                    val_loss += criterion(pred, labels[i].unsqueeze(0)).item()
                    total_pixel_error += compute_pixel_error(pred, labels[i].unsqueeze(0))

        val_loss /= len(val_loader)
        avg_pixel_error = total_pixel_error / len(val_loader.dataset)
        scheduler.step(val_loss)

        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {total_loss/len(train_loader):.4f} | Val Loss: {val_loss:.4f} | Pixel Error: {avg_pixel_error:.2f}px")

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), "model/best_model_tta.pth")
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print("Early stopping")
                break


# ==== Запуск ====
if __name__ == "__main__":
    set_seed(42)
    dataset_path = os.environ.get("DATASET_PATH", "./dataset")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    full_dataset = EyeGazeDataset(dataset_path, transform=get_transform(), return_original=True)
    val_size = int(0.2 * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_set, val_set = random_split(full_dataset, [train_size, val_size])

    # Меняем трансформации на ходу
    train_set.dataset.transform = get_transform(augmented=True)
    val_set.dataset.transform = get_transform(augmented=False)

    train_loader = DataLoader(train_set, batch_size=32, shuffle=True, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, collate_fn=custom_collate_fn)

    model = EyeGazeResNetPix().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()

    print("==== Обучение модели с TTA ====")
    train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=70)
    torch.save(model.state_dict(), "model/model_with_tta.pth")

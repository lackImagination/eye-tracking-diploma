import os
import torch
import pandas as pd
from PIL import Image
from experiments.experiment_6 import EyeGazeNetV3, get_transform

# ==== Настройки ====
# Задай через переменные окружения или измени напрямую для быстрого запуска
dataset_dir = os.environ.get("DATASET_PATH", "./dataset")
model_path = os.environ.get("GAZE_MODEL_PATH")
image_name = os.environ.get("GAZE_IMAGE", "eye_0001.png")

if not model_path:
    raise EnvironmentError("Задай GAZE_MODEL_PATH — путь к .pt файлу модели")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==== Загрузка модели ====
model = torch.jit.load(model_path, map_location=device)
model.eval()

# ==== Предсказание ====
transform = get_transform()
image_path = os.path.join(dataset_dir, "images", image_name)
image = Image.open(image_path).convert("RGB")
input_tensor = transform(image).unsqueeze(0).to(device)

with torch.no_grad():
    predicted_coords = model(input_tensor).cpu().squeeze().numpy()

# ==== Ground truth ====
coords_csv = os.path.join(dataset_dir, "coords.csv")
df = pd.read_csv(coords_csv, header=None, names=["image_name", "x", "y"])
true_coords = df[df["image_name"] == image_name][["x", "y"]].values.squeeze()

print(f"\nImage: {image_name}")
print(f" Ground Truth: {true_coords}")
print(f" Prediction  : {predicted_coords}")
error = ((predicted_coords[0] - true_coords[0])**2 + (predicted_coords[1] - true_coords[1])**2) ** 0.5
print(f" Pixel Error : {error:.2f}px")

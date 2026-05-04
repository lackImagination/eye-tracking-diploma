import torch
import random
import matplotlib
from torch.utils.data import DataLoader, random_split
from screeninfo import get_monitors

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from main import EyeGazeDataset, EyeGazeResNetPix, get_transform, get_tta_transforms, custom_collate_fn, tta_predict

# === Определение экрана — доступно при импорте ===
screen_info = get_monitors()[0]
screen_w, screen_h = screen_info.width, screen_info.height


def denormalize(tensor):
    return tensor.cpu().permute(1, 2, 0).numpy()


def visualize_predictions(model, dataloader, screen_w, screen_h, device, num_samples=5, use_tta=True):
    model.eval()
    all_data = []
    for images, labels, originals in dataloader:
        all_data.append((images, labels, originals))

    indices = random.sample(range(len(all_data)), num_samples)

    for idx in indices:
        images, labels, originals = all_data[idx]
        image = images[0]
        label = labels[0]
        original_img = originals[0]

        with torch.no_grad():
            if use_tta and original_img is not None:
                tta_transforms = get_tta_transforms((224, 224))
                pred = tta_predict(model, original_img, tta_transforms, device)
            else:
                pred = model(image.unsqueeze(0).to(device))

        pred_px = pred.squeeze(0).cpu().numpy()
        actual_px = label.cpu().numpy()

        print(f"Actual (px): {actual_px}, Predicted (px): {pred_px}")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        if original_img is not None:
            ax1.imshow(original_img)
        else:
            ax1.imshow(denormalize(image))
        ax1.scatter(actual_px[0], actual_px[1], color='green', label='Actual', s=100)
        ax1.scatter(pred_px[0], pred_px[1], color='red', label='Predicted', s=100)
        ax1.set_title("Image with Prediction")
        ax1.legend()

        ax2.set_xlim(0, screen_w)
        ax2.set_ylim(screen_h, 0)
        ax2.scatter(actual_px[0], actual_px[1], color='green', label='Actual', s=100)
        ax2.scatter(pred_px[0], pred_px[1], color='red', label='Predicted', s=100)
        ax2.set_title("Screen Coordinates (px)")
        ax2.legend()
        ax2.grid(True)
        ax2.set_xlabel("X")
        ax2.set_ylabel("Y")

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    import os

    dataset_path = os.environ.get("DATASET_PATH", "./dataset")
    model_path = os.environ.get("GAZE_MODEL_PATH", "model/model_with_tta.pth")
    num_images = 10

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = EyeGazeResNetPix().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    transform = get_transform()
    full_dataset = EyeGazeDataset(dataset_path, transform=transform, return_original=True)
    val_size = int(0.2 * len(full_dataset))
    train_size = len(full_dataset) - val_size
    _, val_set = random_split(full_dataset, [train_size, val_size])
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, collate_fn=custom_collate_fn)

    visualize_predictions(model, val_loader, screen_w, screen_h, device, num_samples=num_images, use_tta=True)

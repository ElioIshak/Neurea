import os
import sys
import torch
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from data.dataset import BraTSDataset
from data.split import train_val_split
from utils.load_model import load_model
from metrics.metrics import DiceScore, IoU

def test_neurea():

    load_dotenv()

    DATA_DIR = os.getenv("DATA_PATH")
    SAVED_MODEL_PATH = os.getenv("SAVED_MODEL_PATH")
    SAMPLES = 3

    # Load paths
    paths = []
    for root, dirs, files in os.walk(DATA_DIR):
        for file in files:
            paths.append(os.path.join(root, file))

    _, val_paths = train_val_split(paths, train_split=0.8)

    val_dataset = BraTSDataset(val_paths)

    # Load model
    model = load_model(SAVED_MODEL_PATH)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    dice_fn = DiceScore()
    iou_fn = IoU()

    # Get only slices that contain tumor
    tumor_samples = []

    for i in range(len(val_dataset)):
        image, mask = val_dataset[i]

        if mask.sum() > 0:
            tumor_samples.append((image, mask))

        if len(tumor_samples) == SAMPLES:
            break

    n = len(tumor_samples)

    fig, axes = plt.subplots(
        n,
        3,
        figsize=(18, 5 * n)
    )

    # Column headers
    axes[0, 0].set_title("MRI", fontsize=16, pad=20)
    axes[0, 1].set_title("Ground Truth", fontsize=16, pad=20)
    axes[0, 2].set_title("Prediction", fontsize=16, pad=20)

    for i, (image, mask) in enumerate(tumor_samples):

        with torch.no_grad():
            pred = model(image.unsqueeze(0).to(device))
            pred = torch.sigmoid(pred)
            pred_bin = (pred > 0.5).float()

        mask_gpu = mask.unsqueeze(0).to(device)

        dice = dice_fn(pred_bin, mask_gpu)
        iou = iou_fn(pred_bin, mask_gpu)

        # MRI
        axes[i, 0].imshow(image[0].cpu(), cmap="gray")
        axes[i, 0].axis("off")

        # Ground Truth
        axes[i, 1].imshow(mask[0].cpu(), cmap="gray")
        axes[i, 1].axis("off")

        # Prediction
        axes[i, 2].imshow(pred_bin[0, 0].cpu(), cmap="gray")
        axes[i, 2].axis("off")

        # Metrics shown clearly on the left of each row
        axes[i, 0].set_ylabel(
            f"Sample {i+1}\nDice: {dice:.3f}\nIoU: {iou:.3f}",
            fontsize=12,
            rotation=0,
            labelpad=60,
            va="bottom"
        )

    plt.subplots_adjust(
        left=0.15,
        hspace=0.35,
        wspace=0.15
    )

    plt.show()


if __name__ == "__main__":
    test_neurea()

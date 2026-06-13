# imports
import os
import torch

from dotenv import load_dotenv

from data.dataset import BraTSDataset
from data.split import train_val_split
from losses.losses import DiceLoss
from metrics.metrics import DiceScore, IoU
from models.unet import UNet
from utils.train import train_model
from utils.load_model import load_model

def main():

    load_dotenv()

    BATCH_SIZE = 16
    EPOCHS = 5
    NUM_WORKERS = 10
    PIN_MEMORY = True
    PERSISTENT_WORKERS = True

    DATA_DIR = os.getenv("DATA_PATH")
    SAVED_MODEL_PATH = os.getenv("SAVED_MODEL_PATH")

    # load data
    print("Loading data...")

    paths = []

    for root, dirs, files in os.walk(DATA_DIR):
        for file in files:
            path = os.path.join(root, file)
            paths.append(path)
            print(f"Found file: {file} ; count: {len(paths)}")

    print(f"Found {len(paths)} files.")

    # split dataset
    print("Splitting dataset...")
    train_paths, val_paths = train_val_split(paths, train_split=0.8)

    # create datasets
    print("Creating dataset...")
    train_dataset = BraTSDataset(train_paths)
    val_dataset = BraTSDataset(val_paths)

    # create dataloaders
    print("Creating dataloaders...")
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, persistent_workers=PERSISTENT_WORKERS, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, persistent_workers=PERSISTENT_WORKERS, shuffle=False)

    # initialize model, optimizer, loss function, metrics, and device

    print("Enter 0 for training a new model or 1 for loading an existing model:")
    mode = int(input())

    if mode == 0:
        model = UNet()
    elif mode == 1:
        print(f"Do you want to load the best model in the path {SAVED_MODEL_PATH}? (y/n)")
        choice = input().lower()
        if choice == 'y':
            model = load_model(SAVED_MODEL_PATH)
        else:
            print("Enter the path to the model:")
            model_path = input()
            model = load_model(model_path)

    print("Initializing optimizer...")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    print("Initializing loss function...")
    loss_fn = DiceLoss()    

    print("Initializing metrics...")
    metrics = {
        "dice_score": DiceScore(),
        "iou": IoU()
    }

    print("Checking device...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device} | {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    model.to(device)

    # train model
    print("Training model...")
    history = train_model(model, train_loader, val_loader, optimizer, loss_fn, metrics["dice_score"], metrics["iou"], epochs=EPOCHS, device=device)


if __name__ == "__main__":
    main()
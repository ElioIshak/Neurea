import os
import torch
import tqdm
import pandas as pd

from utils.plot import plot_training_history

def train_model(
        model,
        train_loader,
        val_loader,
        optimizer,
        loss_fn,
        dice_metric,
        iou_metric,
        epochs,
        device='cpu'
):
    
    loss_train_history = []
    dice_train_history = []
    iou_train_history = []

    loss_val_history = []
    dice_val_history = []
    iou_val_history = []

    best_dice = 0

    if os.path.exists("results/history.csv"):
        history_df = pd.read_csv("results/history.csv")
    else:
        history_df = pd.DataFrame(
            columns=[
                "epoch", "train_loss", "val_loss", "train_dice", "val_dice", "train_iou", "val_iou"
            ]
        )

    for epoch in range(epochs):

         # ===== TRAIN ===== #

        model.train()

        train_loss = 0
        train_dice = 0
        train_iou = 0

        train_bar = tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", leave=False)
        for x, y in train_bar:
            x = x.to(device)
            y = y.to(device)
            preds = model(x)

            loss = loss_fn(preds, y)
            train_dice += dice_metric(preds, y) 
            train_iou += iou_metric(preds, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_dice /= len(train_loader)
        train_iou /= len(train_loader)

        loss_train_history.append(train_loss)
        dice_train_history.append(train_dice)
        iou_train_history.append(train_iou)

        train_bar.set_postfix(
            loss=f"{loss.item():.4f}",
            dice=f"{dice_metric(preds, y):.4f}",
            iou=f"{iou_metric(preds, y):.4f}"
        )
        

        # ===== VALIDATION ===== #

        model.eval()

        val_loss = 0
        val_dice = 0
        val_iou = 0

        with torch.no_grad():
            val_bar = tqdm.tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]", leave=False)
            for x, y in val_bar:
                x = x.to(device)
                y = y.to(device)
                preds = model(x)

                loss = loss_fn(preds, y)

                val_loss += loss.item()
                val_dice += dice_metric(preds, y)
                val_iou += iou_metric(preds, y)

        val_loss /= len(val_loader)
        val_dice /= len(val_loader)
        val_iou /= len(val_loader)
        
        loss_val_history.append(val_loss)
        dice_val_history.append(val_dice)
        iou_val_history.append(val_iou)

        val_bar.set_postfix(
            loss=f"{loss.item():.4f}",
            dice=f"{dice_metric(preds, y):.4f}",
            iou=f"{iou_metric(preds, y):.4f}"
        )

        print(f"Epoch: {epoch+1}/{epochs} -> train_loss={train_loss:.4f} | train_dice={train_dice:.4f} | train_iou={train_iou:.4f} | val_loss={val_loss:.4f} | val_dice={val_dice:.4f} | val_iou={val_iou:.4f}")

        new_epoch = pd.DataFrame([{
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_dice": train_dice,
            "val_dice": val_dice,
            "train_iou": train_iou,
            "val_iou": val_iou
        }])

        history_df = pd.concat([history_df, new_epoch], ignore_index=True)
        history_df.to_csv("results/history.csv", index=False)

        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), "best_model.pth")
            print("✅ Best model saved")

    plot_training_history(loss_train_history, loss_val_history, dice_train_history, dice_val_history, iou_train_history, iou_val_history)

    history = {
        'train_loss': loss_train_history,
        'train_dice': dice_train_history,
        'train_iou': iou_train_history,
        'val_loss': loss_val_history,
        'val_dice': dice_val_history,
        'val_iou': iou_val_history,
    }

    return history

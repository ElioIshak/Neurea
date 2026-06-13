import matplotlib.pyplot as plt

def plot_training_history(
    loss_train,
    loss_val,
    dice_train,
    dice_val,
    iou_train, 
    iou_val
):
    epochs = range(1, len(loss_train) + 1)

    plt.figure(figsize=(15, 5))

    # ===== LOSS =====
    plt.subplot(1, 3, 1)
    plt.plot(epochs, loss_train, label="Train Loss")
    plt.plot(epochs, loss_val, label="Val Loss")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    # ===== DICE =====
    plt.subplot(1, 3, 2)
    plt.plot(epochs, dice_train, label="Train Dice")
    plt.plot(epochs, dice_val, label="Val Dice")
    plt.title("Dice Score")
    plt.xlabel("Epoch")
    plt.ylabel("Dice")
    plt.legend()

    # ===== IoU =====
    plt.subplot(1, 3, 3)
    plt.plot(epochs, iou_train, label="Train IoU")
    plt.plot(epochs, iou_val, label="Val IoU")
    plt.title("IoU Score")
    plt.xlabel("Epoch")
    plt.ylabel("IoU")
    plt.legend()

    plt.tight_layout()
    plt.show()
import torch
import torch.nn as nn

class DiceScore(nn.Module):
    def __init__(self, threshold=0.5, smooth=1e-6):
        super().__init__()
        self.smooth = smooth
        self.threshold = threshold

    def __call__(self, preds, target):
        preds = torch.sigmoid(preds)

        preds = (preds > self.threshold).float()

        preds = preds.view(-1)
        target = target.view(-1)

        intersection = (preds * target).sum()

        score = (2 * intersection + self.smooth) / (preds.sum() + target.sum() + self.smooth)

        return score.item()
    

class IoU(nn.Module):
    def __init__(self, threshold=0.5, smooth=1e-6):
        super().__init__()
        self.threshold = threshold
        self.smooth = smooth

    def __call__(self, preds, target):
        preds = torch.sigmoid(preds)

        preds = (preds > self.threshold).float()

        preds = preds.view(-1)
        target = target.view(-1)

        intersection = (preds * target).sum()
        union = preds.sum() + target.sum() - intersection

        iou = (intersection + self.smooth) / (union + self.smooth)

        return iou.item() 
        


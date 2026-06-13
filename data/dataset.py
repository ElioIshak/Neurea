import h5py
import torch

from torch.utils.data import Dataset

class BraTSDataset(Dataset):
    def __init__(self, files_list):
        self.files = files_list

    def __len__(self):
        return len(self.files) 

    def __getitem__(self, index):
        file = self.files[index]

        with h5py.File(file, 'r') as f:
            image = f['image'][:]
            mask = f['mask'][:]

            if image.max() > 0:
                image /= image.max()

            if mask.max() > 0:
                mask = (mask > 0).astype(float)

            image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)
            mask = torch.tensor(mask, dtype=torch.float32).permute(2, 0, 1)

            return image, mask 
       
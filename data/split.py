import random

def train_val_split(files, train_split: float = 0.8):
    # fixed seed
    random.seed(42)

    # dictionary for volumes (vol_1, vol_2, etc.)
    volume_dict = {}
 
    for file in files:
        # split and append based on the volume id 
        id = file.split("_split_")[0]
        if id not in volume_dict:
            volume_dict[id] = []
        volume_dict[id].append(file)

    # volumes
    vol_ids = list(volume_dict.keys())
    random.shuffle(vol_ids) # shuffle the volumes
    
    split = int(train_split * len(vol_ids)) # split index

    # split the volumes to train/val
    train_vols = vol_ids[:split]
    val_vols = vol_ids[split:]

    # group train/val files
    train_files = []
    val_files = []

    for vol in train_vols:
        train_files.extend(volume_dict[vol]) # add all splits of volume_id = vol to train_files

    for vol in val_vols:
        val_files.extend(volume_dict[vol]) # add all splits of volume_id = vol to val_files

    return train_files, val_files
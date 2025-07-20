import torch
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
import os
import re

class LicensePlateDataset(Dataset):
    def __init__(self, root_dir, df, transform=None):
        self.annotations = df.reset_index(drop=True)
        self.root_dir = root_dir
        self.transform = transform

        # Filter: only keep files that exist
        self.annotations = self.annotations[self.annotations['image_name'].apply(
            lambda x: os.path.exists(os.path.join(root_dir, x))
        )].reset_index(drop=True)

        # Build character map
        self.charset = sorted(set(''.join(self.annotations['plate_number'].unique())))
        self.char_to_idx = {char: idx+1 for idx, char in enumerate(self.charset)}
        self.char_to_idx['<blank>'] = 0
        self.idx_to_char = {idx: char for char, idx in self.char_to_idx.items()}
        self.num_chars = len(self.char_to_idx)

    def clean_plate(self, text):
        return re.sub(r"[^A-Z0-9]", "", text.upper())

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        try:
            row = self.annotations.iloc[idx]
            img_path = os.path.join(self.root_dir, row["image_name"])
            plate_number = self.clean_plate(row["plate_number"])

            if len(plate_number) == 0:
                return self.__getitem__((idx + 1) % len(self))

            image = Image.open(img_path).convert("L")
            if self.transform:
                image = self.transform(image)

            target = torch.tensor([self.char_to_idx[c] for c in plate_number], dtype=torch.long)
            return image, target, plate_number

        except Exception as e:
            print(f"⚠️ Error at idx {idx}: {e}")
            return self.__getitem__((idx + 1) % len(self))

    def get_vocab_size(self):
        return self.num_chars

    def decode_plate(self, indices):
        return ''.join([self.idx_to_char.get(idx.item(), "") for idx in indices])

def decode(pred_tensor, idx_to_char):
    prev = -1
    result = []
    for idx in pred_tensor:
        idx = idx.item()
        if idx != prev and idx != 0:
            result.append(idx_to_char.get(idx, ""))
        prev = idx
    return ''.join(result)

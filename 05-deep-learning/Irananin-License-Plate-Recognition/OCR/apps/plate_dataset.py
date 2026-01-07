import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset


IMG_HEIGHT = 50
IMG_WIDTH = 200


MAX_LABEL_LEN = 8
PAD_CHAR = "_"

class PlateDataset(Dataset):
    def __init__(self, root_dir, mode="train"):
        self.samples = []
        self.root_dir = root_dir

        label_path = os.path.join(root_dir, f"{mode}_labels_fa.txt")
        with open(label_path, "r", encoding="utf-8") as f:
            for line in f:
                img, text = line.strip().split()
                if len(text) > MAX_LABEL_LEN:
                    continue
                text = text.ljust(MAX_LABEL_LEN, PAD_CHAR)
                self.samples.append((img, text))


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_rel_path, text = self.samples[idx]
        img_path = os.path.join(self.root_dir, img_rel_path)

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
        img = img.astype(np.float32) / 255.0
        img = (img - 0.5) / 0.5
        img = np.expand_dims(img, 0)

        return torch.from_numpy(img), text

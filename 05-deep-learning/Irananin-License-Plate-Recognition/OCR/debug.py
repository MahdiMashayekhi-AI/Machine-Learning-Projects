import os
import cv2
import json
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from apps.plate_dataset import PlateDataset
from apps.label_convertor import LabelConvertor

# ---------------- CONFIG ----------------
IMG_HEIGHT = 48
IMG_WIDTH = 160
BATCH_SIZE = 4
DATASET_DIR = "./dataset"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------- LOAD CLASSES ----------------
with open("./classes.json", "r", encoding="utf-8") as f:
    CLASSES_LIST = json.load(f)["classes"]

# ---------------- MODEL ----------------
class CRNN(nn.Module):
    def __init__(self, nc, nclass, nh=256):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(nc, 64, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d((2, 2), (2, 1)),
            nn.Conv2d(256, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.MaxPool2d((2, 2), (2, 1)),
            nn.Conv2d(512, 512, 2, 1, 0), nn.BatchNorm2d(512), nn.ReLU()
        )

        self.rnn = nn.LSTM(
            input_size=512,
            hidden_size=nh,
            num_layers=2,
            bidirectional=True,
            batch_first=True
        )

        self.fc = nn.Linear(nh * 2, nclass)

    def forward(self, x):
        x = self.cnn(x)
        print("CNN output:", x.shape)  # [B, C, H, W]

        x = F.adaptive_avg_pool2d(x, (1, x.size(3)))
        print("After adaptive pool:", x.shape)

        x = x.squeeze(2).permute(0, 2, 1)
        print("Sequence shape:", x.shape)  # [B, T, C]

        x, _ = self.rnn(x)
        x = self.fc(x)
        return x


# ---------------- DEBUG PIPELINE ----------------
def main():
    print("\n==== DATASET CHECK ====")
    train_ds = PlateDataset(DATASET_DIR, "train")
    print("Total samples:", len(train_ds))

    img, text = train_ds[0]
    print("Single sample image shape:", img.shape)
    print("Single label:", text)

    print("\n==== LABEL ENCODING CHECK ====")
    converter = LabelConvertor(CLASSES_LIST)
    targets, lengths = converter.encode([text])
    print("Encoded:", targets.tolist())
    print("Length:", lengths.tolist())

    print("\n==== DATALOADER CHECK ====")
    loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    imgs, texts = next(iter(loader))
    print("Batch image shape:", imgs.shape)
    print("Batch texts:", texts)

    imgs = imgs.to(DEVICE)

    print("\n==== MODEL FORWARD CHECK ====")
    model = CRNN(nc=1, nclass=len(CLASSES_LIST) + 1).to(DEVICE)
    with torch.no_grad():
        out = model(imgs)

    print("\n==== OUTPUT CHECK ====")
    print("Raw output shape:", out.shape)  # [B, T, C]
    T = out.size(1)
    max_label_len = max(len(t) for t in texts)

    print(f"T (time steps): {T}")
    print(f"Max label length: {max_label_len}")
    print(f"T / label ratio: {T / max_label_len:.2f}")

    print("\n==== BLANK PROBABILITY CHECK ====")
    probs = out.softmax(2)
    blank_prob = probs[:, :, 0].mean().item()
    print(f"Mean blank probability: {blank_prob:.4f}")

    print("\n==== SANITY CHECK ====")
    argmax_seq = probs.argmax(2)[0].cpu().numpy()
    decoded = []
    prev = 0
    for k in argmax_seq:
        if k != 0 and k != prev:
            decoded.append(converter.id2cls[k])
        prev = k
    print("Greedy decode:", "".join(decoded))


if __name__ == "__main__":
    main()

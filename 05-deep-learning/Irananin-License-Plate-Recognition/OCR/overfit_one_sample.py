import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from apps.plate_dataset import PlateDataset
from apps.label_convertor import LabelConvertor

# ================= CONFIG =================
IMG_HEIGHT = 48
IMG_WIDTH = 160
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LR = 1e-3
EPOCHS = 500

# ============== LOAD CLASSES ==============
with open("classes.json", "r", encoding="utf-8") as f:
    CLASSES = json.load(f)["classes"]

# ================ MODEL ===================
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
            512, nh, num_layers=2,
            bidirectional=True, batch_first=True
        )

        self.fc = nn.Linear(nh * 2, nclass)

        self.fc.bias.data[0] = -2.0

    def forward(self, x):
        x = self.cnn(x)
        x = F.adaptive_avg_pool2d(x, (1, x.size(3)))
        x = x.squeeze(2).permute(0, 2, 1)
        x, _ = self.rnn(x)
        x = self.fc(x)
        return x

# =============== DECODE ===================
def greedy_decode(logits, converter):
    probs = logits.softmax(2)
    seq = probs.argmax(2)[0].cpu().numpy()
    res = []
    prev = 0
    for k in seq:
        if k != 0 and k != prev:
            res.append(converter.id2cls[k])
        prev = k
    return "".join(res)

# ================= MAIN ===================
def main():
    print("=== OVERFIT ONE SAMPLE TEST ===")

    dataset = PlateDataset("./dataset", "train")

    dataset.samples = dataset.samples[:1]

    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    img, text = next(iter(loader))
    img = img.to(DEVICE)

    print("GT LABEL:", text[0])

    converter = LabelConvertor(CLASSES)
    targets, lengths = converter.encode(text)

    model = CRNN(1, len(CLASSES) + 1).to(DEVICE)

    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(1, EPOCHS + 1):
        model.train()

        logits = model(img)
        log_probs = logits.log_softmax(2).permute(1, 0, 2)
        input_lengths = torch.tensor([log_probs.size(0)])

        loss = criterion(log_probs, targets, input_lengths, lengths)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0 or epoch <= 20:
            model.eval()
            with torch.no_grad():
                pred = greedy_decode(logits, converter)
            print(f"Epoch {epoch:03d} | Loss {loss.item():.4f} | PR: {pred}")

    print("=== END ===")

if __name__ == "__main__":
    main()

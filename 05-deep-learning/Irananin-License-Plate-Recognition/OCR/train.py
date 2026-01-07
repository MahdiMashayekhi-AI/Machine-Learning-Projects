import os
import cv2
import json
import csv
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from apps.plate_dataset import PlateDataset
from apps.label_convertor import LabelConvertor

# ================= CONFIG =================
BATCH_SIZE = 64
EPOCHS = 25
LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATASET_DIR = "./dataset"
LOG_PATH = "./logs/results2.csv"
BEST_MODEL_PATH = "./models/best_crnn_model_v2.pth"
LAST_MODEL_PATH = "./models/last_crnn_model_v2.pth"

os.makedirs("./models", exist_ok=True)
os.makedirs("./logs", exist_ok=True)

with open("classes.json", "r", encoding="utf-8") as f:
    CLASSES_LIST = json.load(f)['classes']

# ================= MODEL ===================
class CRNN(nn.Module):
    def __init__(self, nc, nclass, nh=256):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(nc, 64, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2, 2), nn.Dropout(0.2),
            nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2, 2),nn.Dropout(0.2),
            nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d((2, 2), (2, 1)), nn.Dropout(0.2),
            nn.Conv2d(256, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.MaxPool2d((2, 2), (2, 1)), nn.Dropout(0.2),
            nn.Conv2d(512, 512, 2, 1, 0), nn.BatchNorm2d(512), nn.ReLU()
        )

        self.rnn = nn.LSTM(
            512, nh, num_layers=2,
            bidirectional=True, batch_first=True, dropout=0.3
        )

        self.dropout = nn.Dropout(0.2)

        self.fc = nn.Linear(nh * 2, nclass)

        self.fc.bias.data[0] = -2.0

    def forward(self, x):
        x = self.cnn(x)
        x = F.adaptive_avg_pool2d(x, (1, x.size(3)))
        x = x.squeeze(2).permute(0, 2, 1)
        x, _ = self.rnn(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight.data)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

# ================= VALIDATION =================
def validate(model, loader, converter, criterion):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for imgs, texts in loader:
            imgs = imgs.to(DEVICE)
            targets, lengths = converter.encode(texts)
            logits = model(imgs)
            log_probs = logits.log_softmax(2).permute(1, 0, 2)
            input_lengths = torch.full((logits.size(0),), log_probs.size(0), dtype=torch.long)
            
            loss = criterion(log_probs, targets, input_lengths, lengths)
            val_loss += loss.item()
    
    pred_text = converter.decode(logits.log_softmax(2).permute(1, 0, 2))[0]
    return val_loss / len(loader), texts[0], pred_text

# ================= TRAINING =================
def train():
    converter = LabelConvertor(CLASSES_LIST)
    train_ds = PlateDataset(DATASET_DIR, "train")
    test_ds = PlateDataset(DATASET_DIR, "test")
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = CRNN(1, len(CLASSES_LIST) + 1).to(DEVICE)
    model.apply(weights_init)
    
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-5)

    best_val_loss = float('inf')
    
    with open(LOG_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Epoch", "Train_Loss", "Val_Loss"])

    print(f"🚀 Training started on {DEVICE}...")
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        
        # Progress Bar
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for imgs, texts in pbar:
            imgs = imgs.to(DEVICE)

            # Encode labels (padding حذف می‌شود داخل converter)
            targets, lengths = converter.encode(texts)
            targets = targets.to(DEVICE)
            lengths = lengths.to(DEVICE)

            optimizer.zero_grad()

            # Forward
            logits = model(imgs)              # [B, T, C]
            log_probs = logits.log_softmax(2) # [B, T, C]
            log_probs = log_probs.permute(1, 0, 2)  # [T, B, C]

            # === CTC input lengths (ثابت و درست) ===
            T = log_probs.size(0)
            input_lengths = torch.full(
                size=(imgs.size(0),),
                fill_value=T,
                dtype=torch.long,
                device=DEVICE
            )

            # Loss
            loss = criterion(log_probs, targets, input_lengths, lengths)

            # Backward
            loss.backward()

            # Gradient clipping 
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

            optimizer.step()

            train_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")


        # Validation
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss, gt_sample, pr_sample = validate(model, test_loader, converter, criterion)
        
        scheduler.step(avg_val_loss)

        print(f"End Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        print(f"Sample -> GT: {gt_sample} | PR: {pr_sample}")

        with open(LOG_PATH, "a", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch+1, avg_train_loss, avg_val_loss])

        torch.save(model.state_dict(), LAST_MODEL_PATH)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"Best Model Saved (Val Loss: {best_val_loss:.4f})")

if __name__ == "__main__":
    train()
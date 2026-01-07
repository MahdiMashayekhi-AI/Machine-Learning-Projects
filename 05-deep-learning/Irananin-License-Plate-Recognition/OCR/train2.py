import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn import CTCLoss
import numpy as np
import os
import cv2
from apps.label_convertor import LabelConvertor

# --- تنظیمات ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATASET_DIR = "./dataset" # مسیر دیتاست

# --- 1. دیتالودر دستی (برای اطمینان از رفع باگ imgae) ---
def get_debug_batch(batch_size=8):
    samples = []
    # خواندن فایل لیبل
    with open(f"{DATASET_DIR}/train_labels.txt", "r", encoding="utf-8") as f:
        lines = f.readlines()[:batch_size] # فقط ۸ تا اول
    
    images = []
    texts = []
    
    print("\n🔍 CHECKING INPUT DATA:")
    for i, line in enumerate(lines):
        parts = line.strip().split(' ')
        img_path, text = parts[0], parts[1]
        
        # خواندن عکس
        full_path = os.path.join(DATASET_DIR, img_path)
        img = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            print(f"❌ Error: Could not read image {full_path}")
            continue
            
        # --- نمایش سایز قبل و بعد از ریسایز ---
        if i == 0:
            print(f"   Original Size: {img.shape}")
            
        # ریسایز صحیح
        img = cv2.resize(img, (160, 48)) 
        
        if i == 0:
            print(f"   Resized Size: {img.shape} (Must be 48, 160)")
            print(f"   Label: {text}")
            
        img = img.astype(np.float32) / 255.0
        images.append(img)
        texts.append(text)
        
    # تبدیل به تنسور
    images = np.array(images)
    images = np.expand_dims(images, axis=1) # (Batch, 1, 48, 160)
    return torch.from_numpy(images).float().to(DEVICE), texts

# --- 2. مدل Baby CRNN (خیلی ساده و سبک برای تست) ---
class BabyCRNN(nn.Module):
    def __init__(self, nclass):
        super().__init__()
        # فقط 3 لایه کانولوشن برای تست سلامت دیتا
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2, 2), # -> 24x80
            nn.Conv2d(32, 64, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2, 2), # -> 12x40
            nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(), nn.MaxPool2d((2, 1)), # -> 6x40
        )
        self.rnn = nn.LSTM(128, 64, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(128, nclass)

    def forward(self, x):
        # x: (B, 1, 48, 160)
        x = self.cnn(x) 
        # میانگین گیری روی ارتفاع باقی‌مانده (6 پیکسل)
        x = x.mean(dim=2) # -> (B, 128, 40)
        x = x.permute(0, 2, 1) # -> (B, 40, 128)
        x, _ = self.rnn(x)
        x = self.fc(x)
        return x

# --- 3. اجرای تست ---
def run_debug_test():
    # آماده سازی کلاس ها
    import json
    with open("./classes.json", "r") as f:
        CLASSES = json.load(f)['classes']
    
    converter = LabelConvertor(CLASSES)
    model = BabyCRNN(nclass=len(CLASSES)+1).to(DEVICE)
    
    # استفاده از RMSprop و نرخ یادگیری کمتر
    optimizer = optim.RMSprop(model.parameters(), lr=0.0005)
    criterion = CTCLoss(blank=0, zero_infinity=True)
    
    # دریافت دیتا
    images, texts = get_debug_batch(8)
    targets, target_lengths = converter.encode(texts)
    
    print("\n🚀 Starting Baby Model Training...")
    model.train()
    
    for i in range(201):
        optimizer.zero_grad()
        preds = model(images) # (B, Time, Class)
        preds = preds.log_softmax(2).permute(1, 0, 2) # (Time, B, Class)
        
        input_lengths = torch.full(size=(images.size(0),), fill_value=preds.size(0), dtype=torch.long)
        
        loss = criterion(preds, targets, input_lengths, target_lengths)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5) # کلیپ گرادیان
        optimizer.step()
        
        if i % 20 == 0:
            # دیکود کردن
            with torch.no_grad():
                pred_idx = preds[:, 0, :].argmax(1)
                res = ""
                prev = 0
                for k in pred_idx:
                    k = k.item()
                    if k != 0 and k != prev:
                        res += converter.id2cls[k]
                    prev = k
            print(f"Iter {i}: Loss={loss.item():.4f} | Pred: {res}")
            
            if loss.item() < 0.1:
                print("\n✅ SUCCESS! System is working.")
                return
            
    # این تست را روی همان 8 عکس انجام بده
    converter = LabelConvertor(CLASSES)
    text = "55SH12318"
    encoded, length = converter.encode([text])

    print(f"Original Text: {text}")
    print(f"Encoded IDs: {encoded.tolist()}")
    print(f"Decoded back: {[converter.id2cls[i] for i in encoded.tolist()]}")
    print(f"Encoded Length: {length.item()}")

if __name__ == "__main__":
    run_debug_test()
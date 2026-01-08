import os
import cv2
import json
import torch
import numpy as np
from ultralytics import YOLO
from model import CRNN
from converter import LabelConvertor

# ================= CONFIG =================
IMG_HEIGHT = 50
IMG_WIDTH = 200

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DETECTION_MODEL_PATH = "./models/detection_model.pt"
OCR_MODEL_PATH = "./models/best_crnn_model.pth"
TEST_IMAGE = "./data/2.jpg"

# ================= LOAD CLASSES =================
with open("classes.json", "r", encoding="utf-8") as f:
    CLASSES_LIST = json.load(f)["classes"]


ocr_model = CRNN(1, len(CLASSES_LIST) + 1).to(DEVICE)
ocr_model.load_state_dict(torch.load(OCR_MODEL_PATH, map_location=DEVICE))
ocr_model.eval()

converter = LabelConvertor(CLASSES_LIST)

# ================= DETECTION MODEL =================
detector = YOLO(DETECTION_MODEL_PATH)

# ================= UTILS =================
def resize_with_padding(gray, target_w=200, target_h=50):
    h, w = gray.shape
    scale = min(target_w / w, target_h / h)
    nw, nh = int(w * scale), int(h * scale)

    resized = cv2.resize(gray, (nw, nh))
    canvas = np.zeros((target_h, target_w), dtype=np.uint8)

    x = (target_w - nw) // 2
    y = (target_h - nh) // 2
    canvas[y:y+nh, x:x+nw] = resized
    return canvas


def preprocess_plate(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (IMG_WIDTH, IMG_HEIGHT))

    gray = gray.astype(np.float32) / 255.0
    gray = (gray - 0.5) / 0.5

    tensor = torch.from_numpy(gray).unsqueeze(0).unsqueeze(0)
    return tensor.to(DEVICE), gray


def ocr_predict(plate_img):
    x, debug_img = preprocess_plate(plate_img)

    # sanity check
    # cv2.imshow("ocr_input", debug_img)
    # cv2.waitKey(0)

    with torch.no_grad():
        logits = ocr_model(x)
        log_probs = logits.log_softmax(2).permute(1, 0, 2)
        text = converter.decode(log_probs)[0]

    return text


# ================= MAIN PIPELINE =================
def recognize_plates(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(image_path)

    results = detector(img, conf=0.4)

    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        print("No plate detected")
        return []

    h, w, _ = img.shape
    pad = 5

    outputs = []

    for i, box in enumerate(boxes.xyxy):
        x1, y1, x2, y2 = box.cpu().numpy().astype(int)

        # expand box
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(w, x2 + pad)
        y2 = min(h, y2 + pad)

        plate_crop = img[y1:y2, x1:x2]

        plate_text = ocr_predict(plate_crop)

        outputs.append({
            "index": i,
            "text": plate_text,
            "bbox": (x1, y1, x2, y2)
        })

        # cv2.imwrite(f"./data/plate_{i}_{plate_text}.jpg", plate_crop)
        print(f"✅ Plate {i}: {plate_text}")

    return outputs


# ================= RUN =================
if __name__ == "__main__":
    recognize_plates(TEST_IMAGE)

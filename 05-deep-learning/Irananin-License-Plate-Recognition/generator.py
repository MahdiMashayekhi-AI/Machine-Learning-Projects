import os
import cv2
import json
import numpy as np
from glob import glob
from pathlib import Path
import torch
import os
from configs.config_loader import cfg
from core.pipeline import Pipeline
from core.detection.inference.predictor import DetPredictor
from core.detection.models.yolo_detector import YoloDetector
from core.ocr.models.crnn import CRNN
from core.ocr.data.label_converter import LabelConverter
from core.ocr.inference.predictor import OCRPredictor
from core.visualization.drawer import Drawer

IMAGE_PATH = "./data/detection/samples/2.jpg"
DET_MODEL_PATH = cfg["detection"]["model_path"]
OCR_MODEL_PATH = cfg["ocr"]["model_path"]
CLASSES_PATH = cfg["ocr"]["classes_path"]
DEVICE = ("cuda" if torch.cuda.is_available() else "cpu")

with open(CLASSES_PATH, "r", encoding="utf-8") as f:
  classes = json.load(f)["classes"]

# Detection
detector = YoloDetector(DET_MODEL_PATH, DEVICE)
det = DetPredictor(detector)

# OCR
model = CRNN(1, len(classes)+1).to(DEVICE)
model.load_state_dict(torch.load(OCR_MODEL_PATH, map_location=DEVICE)["model"])

convertor = LabelConverter(classes)
ocr = OCRPredictor(model, convertor, DEVICE)

# ---------------- CONFIG ----------------
OUTPUT_DIR = "./data/ocr_dataset/train/images"
LABEL_FILE = "./data/ocr_dataset/train/labels.txt"
START_INDEX = 35705 # اگر None باشد خودش حساب می‌کند
AUG_PER_IMAGE = 5
LABEL_JSON_PATH = "./data/detection/raw/test/test_labels.json"
IMAGE_DIR = "./data/detection/raw/test/images"
# ----------------------------------------

def get_next_index(output_dir):
    files = glob(os.path.join(output_dir, "*.jpg"))
    if not files:
        return 1

    nums = []
    for f in files:
        name = os.path.basename(f)
        num = name.split("_")[0]
        if num.isdigit():
            nums.append(int(num))
    return max(nums) + 1 if nums else 1


# ---------------- AUGMENTATIONS (REALISTIC & CONTROLLED) ----------------

def realistic_motion_blur(img):
    k = np.random.choice([3, 5])  # کوچک و کنترل‌شده
    kernel = np.zeros((k, k))
    kernel[int((k-1)/2), :] = np.ones(k)
    kernel = kernel / k
    return cv2.filter2D(img, -1, kernel)


def realistic_gaussian_blur(img):
    return cv2.GaussianBlur(img, (3, 3), sigmaX=0.8)


def realistic_brightness_contrast(img):
    alpha = np.random.uniform(0.9, 1.1)   # contrast
    beta = np.random.uniform(-15, 15)     # brightness
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)


def realistic_perspective_rotate(img):
    h, w = img.shape[:2]

    # Rotation محدود و واقعی
    angle = np.random.uniform(-12, 12)
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    rotated = cv2.warpAffine(
        img, M, (w, h),
        borderMode=cv2.BORDER_REPLICATE
    )

    # Perspective ملایم
    max_shift = 0.05 * w  # فقط 5 درصد

    pts1 = np.float32([[0,0],[w,0],[0,h],[w,h]])
    pts2 = np.float32([
        [np.random.uniform(0, max_shift), np.random.uniform(0, max_shift)],
        [w - np.random.uniform(0, max_shift), np.random.uniform(0, max_shift)],
        [np.random.uniform(0, max_shift), h - np.random.uniform(0, max_shift)],
        [w - np.random.uniform(0, max_shift), h - np.random.uniform(0, max_shift)]
    ])

    matrix = cv2.getPerspectiveTransform(pts1, pts2)

    warped = cv2.warpPerspective(
        rotated,
        matrix,
        (w, h),
        borderMode=cv2.BORDER_REPLICATE
    )

    return warped


def realistic_noise(img):
    img_float = img.astype(np.float32)

    std = np.random.uniform(3, 6)  # خیلی کمتر از قبل
    noise = np.random.normal(0, std, img.shape).astype(np.float32)

    noisy = img_float + noise
    noisy = np.clip(noisy, 0, 255)

    return noisy.astype(np.uint8)


def apply_random_augmentation(img):
    aug_functions = [
        realistic_motion_blur,
        realistic_gaussian_blur,
        realistic_brightness_contrast,
        realistic_perspective_rotate,
        realistic_noise
    ]

    # فقط یکی یا دو تا اعمال بشه نه همه
    num_aug = np.random.choice([1, 2])
    chosen = np.random.choice(aug_functions, num_aug, replace=False)

    augmented = img.copy()
    for func in chosen:
        augmented = func(augmented)

    return augmented

# ---------------- MAIN PROCESS ----------------

def build_dataset(label_json_path, image_dir, pipeline):

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(label_json_path, "r", encoding="utf-8") as f:
        gt_labels = json.load(f)

    index = get_next_index(OUTPUT_DIR) if START_INDEX is None else START_INDEX

    label_lines = []

    for img_name, gt_text in gt_labels.items():

        if gt_text == "unknown":
            continue

        img_path = os.path.join(image_dir, img_name)
        frame = cv2.imread(img_path)
        if frame is None:
            continue

        results = pipeline.process_frame(frame)
        if len(results) == 0:
            continue

        # بهترین detection
        bbox = results[0].bbox
        crop = pipeline._cropper(frame, bbox)

        if crop is None:
            continue

        # --- Original ---
        filename = f"{index}_{gt_text}.jpg"
        save_path = os.path.join(OUTPUT_DIR, filename)
        cv2.imwrite(save_path, crop)

        label_lines.append(f"train/{filename} {gt_text}")
        index += 1

        for _ in range(AUG_PER_IMAGE):
          aug_img = apply_random_augmentation(crop)

          filename = f"{index}_{gt_text}.jpg"
          save_path = os.path.join(OUTPUT_DIR, filename)
          cv2.imwrite(save_path, aug_img)

          label_lines.append(f"train/{filename} {gt_text}")
          index += 1

    # نوشتن فایل لیبل
    with open(LABEL_FILE, "a", encoding="utf-8") as f:
        for line in label_lines:
            f.write(line + "\n")

    print("Dataset building finished.")

pipeline = Pipeline(det, ocr)
build_dataset(LABEL_JSON_PATH, IMAGE_DIR, pipeline)
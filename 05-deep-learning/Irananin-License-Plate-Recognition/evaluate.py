import json
import cv2
import torch
import os
from core.pipeline import Pipeline
from core.ocr.utils.metrics import cer, sequence_accuracy 
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

def evaluate_on_real_data(label_json_path, image_dir, pipeline):
    with open(label_json_path, 'r', encoding='utf-8') as f:
        gt_labels = json.load(f)

    preds_list = []
    targets_list = []
    
    print("Starting Evaluation...")
    
    for img_name, target_text in gt_labels.items():
        if target_text == "unknown":
            continue
            
        img_path = os.path.join(image_dir, img_name)
        frame = cv2.imread(img_path)
        
        if frame is None:
            continue
            
        # اجرای پایپ‌لاین
        results = pipeline.process_frame(frame)
        
        # اگر پلاک پیدا شد، اولین پلاک رو بردار
        if len(results) > 0:
            pred_text = results[0].text
        else:
            pred_text = "" # پلاک پیدا نشد
            
        preds_list.append(pred_text)
        targets_list.append(target_text)
        
        if target_text != pred_text:
          print(f"File: {img_name} | GT: {target_text} | Pred: {pred_text}")

    # محاسبه مترییک‌ها
    final_cer = cer(preds_list, targets_list)
    final_acc = sequence_accuracy(preds_list, targets_list)
    
    print("\n" + "="*30)
    print(f"📊 Final Results on Real Data:")
    print(f"Total Images: {len(targets_list)}")
    print(f"Character Error Rate (CER): {final_cer:.4f}")
    print(f"Sequence Accuracy: {final_acc * 100:.2f}%")
    print("="*30)

labels_apth = "./data/detection/raw/test/test_labels.json"
images_dir = "./data/detection/raw/test/images"
pipeline = Pipeline(det, ocr)
evaluate_on_real_data(labels_apth, images_dir, pipeline)
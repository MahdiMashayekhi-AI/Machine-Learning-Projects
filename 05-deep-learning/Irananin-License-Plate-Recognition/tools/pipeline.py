import cv2
import json
import torch
from configs import ocr
from core.pipeline import Pipeline
from core.detection.inference.predictor import Predictor as DetPredictor
from core.detection.models.yolo_detector import YoloDetector
from core.ocr.models.crnn import CRNN
from core.ocr.data.label_converter import LabelConverter
from core.ocr.inference.predictor import Predictor as OCRPredictor

IMAGE_PATH = "./data/detection/samples/image-1.jpg"
DET_MODEL_PATH = "./models/detection/best.pt"
OCR_MODEL_PATH = "./models/ocr/best.pth"
CLASSES_PATH = ocr.CLASSES_PATH
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

image = cv2.imread(IMAGE_PATH)

pipeline = Pipeline(det=det, ocr=ocr, threshold=0.25)
print(pipeline.run(image))
import os
import cv2  
import json
import torch
from configs.config_loader import cfg
from core.ocr.models.crnn import CRNN
from core.ocr.data.label_converter import LabelConverter
from core.ocr.data.transforms import preprocess_image
from core.ocr.inference.predictor import OCRPredictor
from core.ocr.inference.postprocess import PlateValidator


MODEL_PATH = cfg["ocr"]["model_path"]
CLASSES_PATH = cfg["ocr"]["classes_path"]
DEVICE = ("cuda" if torch.cuda.is_available() else "cpu")


with open(CLASSES_PATH, "r", encoding="utf-8") as f:
  classes = json.load(f)["classes"]


model = CRNN(1, len(classes)+1).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE)["model"])

convertor = LabelConverter(classes)

predictor = OCRPredictor(model, convertor, DEVICE)

validator = PlateValidator()

def run_inference(image):
  result = predictor.predict(image)
  return validator.normalize(result.text)


if __name__ == "__main__":
  img_path = "./data/ocr/samples/1_plate.jpg"
  image = cv2.imread(img_path)
  print(run_inference(image))
import io
import cv2
import json
import torch
import numpy as np
from PIL import Image
from configs.config_loader import cfg
from core.pipeline import Pipeline
from core.detection.inference.predictor import DetPredictor
from core.detection.models.yolo_detector import YoloDetector
from core.ocr.models.crnn import CRNN
from core.ocr.data.label_converter import LabelConverter
from core.ocr.inference.predictor import OCRPredictor
from fastapi import FastAPI, UploadFile, HTTPException, File 
from fastapi.middleware.cors import CORSMiddleware
from core.pipeline import Pipeline
from core.utils.mapping import map_plate
from pydantic import BaseModel
from typing import List


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

# Pipeline
pipeline = Pipeline(det, ocr)


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PlateDetectionResponse(BaseModel):
    bbox: List[int]
    text: str
    det_conf: float
    ocr_conf: float

class FinalResponse(BaseModel):
    status: str
    data: List[PlateDetectionResponse]

@app.get("/")
def home():
  return {"message": "AI Plate Recognition API is Online!"}


@app.post('/predict', response_model=FinalResponse)
async def predict(file: UploadFile = File(...)):
  try:
    image = await file.read()
    image = Image.open(io.BytesIO(image)).convert("RGB")
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
  except Exception:
     raise HTTPException(status_code=400, detail="Invalid file!")

  results = pipeline.process_frame(image)

  outputs = []
  for result in results:
    outputs.append({
      "bbox": result.bbox,
      "text": map_plate(result.text),
      "det_conf": round(result.det_conf * 100, 2),
      "ocr_conf": round(result.ocr_conf * 100, 2)
    })

  return {"status": "success", "data": outputs}
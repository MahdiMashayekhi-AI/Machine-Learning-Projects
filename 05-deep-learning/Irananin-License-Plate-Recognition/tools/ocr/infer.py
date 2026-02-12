import os
import json
import torch
from configs import ocr
from core.ocr.models.crnn import CRNN
from core.ocr.data.lablel_convertor import LabelConvertor
from core.ocr.data.transforms import preprocess_image
from core.ocr.inference.predictor import Predictor
from core.ocr.inference.postprocess import postprocess


MODEL_PATH = os.path.join(ocr.EXPERIMENT_ROOT, "exp_001/checkpoints/best.pth")
CLASSES_PATH = ocr.CLASSES_PATH
DEVICE = ocr.DEVICE


with open(CLASSES_PATH, "r", encoding="utf-8") as f:
  classes = json.load(f)["classes"]


model = CRNN(1, len(classes)+1).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE)["model"])

convertor = LabelConvertor(classes)

predictor = Predictor(model, convertor, DEVICE)


def run_inference(img_path):
  image = preprocess_image(img_path)
  image = image.unsqueeze(0)

  preds = predictor.predict(image)
  preds = postprocess(preds)

  return preds[0]


if __name__ == "__main__":
  img = "./data/ocr/samples/8_plate.jpg"
  print(run_inference(img))
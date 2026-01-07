import os
import cv2
import json
import torch
import numpy as np
from train import CRNN
from apps.label_convertor import LabelConvertor


IMG_HEIGHT = 50
IMG_WIDTH = 200
MODEL_PATH = "./models/best_crnn_model_v2.pth"
TEST_IMG = "./data/10_plate.jpg"
DEVICE = ("cuda" if torch.cuda.is_available() else "cpu")

with open("./classes.json", "r", encoding="utf-8") as f:
  CLASSES_LIST = json.load(f)['classes']


def predict():
  # Loading the crnn model
  model = CRNN(1, len(CLASSES_LIST) + 1).to(DEVICE)
  model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))

  # Loading the test image
  img = cv2.imread(TEST_IMG, cv2.IMREAD_GRAYSCALE)
  img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
  img = img.astype(np.float32) / 255.0
  img = (img - 0.5) / 0.5
  cv2.imwrite("./data/6_plate_resized.jpg" , img * 255)
  img = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)


  # Running the model on the test image
  with torch.no_grad():
    model.eval()

    img = img.to(DEVICE)
    logits = model(img).to(DEVICE)
    log_probs = logits.log_softmax(2).permute(1, 0, 2)

  # Decoding the predicted text
  converter = LabelConvertor(CLASSES_LIST)
  pred_text = converter.decode(log_probs)[0]

  return pred_text


if __name__ == "__main__":  
  print(predict())


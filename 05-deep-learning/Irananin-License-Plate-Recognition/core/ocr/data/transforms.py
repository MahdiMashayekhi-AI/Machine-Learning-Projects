import cv2
import torch
import numpy as np
from configs import ocr 

IMAGE_SIZE = ocr.IMAGE_SIZE

def preprocess_image(img_path):
  img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
  img = cv2.resize(img, IMAGE_SIZE)
  img = img / 255.0
  img = (img - 0.5) / 0.5
  img = np.expand_dims(img, 0)
  return torch.from_numpy(img).to(torch.float32)
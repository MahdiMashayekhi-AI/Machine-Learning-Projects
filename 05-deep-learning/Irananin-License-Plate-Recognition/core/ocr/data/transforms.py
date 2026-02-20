import cv2
import torch
import numpy as np
from configs.config_loader import cfg

IMAGE_SIZE = cfg['ocr']['image_size']

def preprocess_image(image):
  if len(image.shape) == 3:
        img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  else:
      img = image.copy()

  img = cv2.resize(img, IMAGE_SIZE)
  img = img / 255.0
  img = (img - 0.5) / 0.5
  img = np.expand_dims(img, 0)
  img = np.expand_dims(img, 0)
  return torch.from_numpy(img).to(torch.float32)
import torch
import numpy as np
from core.entities.ocr import OCRResult
from core.ocr.data.transforms import preprocess_image

class Predictor:
  def __init__(self, model, convertor, device):
    self.model = model
    self.convertor = convertor
    self.device = device
    self.model.to(device)
    self.model.eval()


  @torch.no_grad()
  def predict(self, image: np.ndarray) -> OCRResult:
    input_tensor = preprocess_image(image).to(self.device)

    logits = self.model(input_tensor)
    texts = self.convertor.decode(logits)
    text = texts[0] if texts else ""

    return OCRResult(
      text=text,
      conf=None
    )
    

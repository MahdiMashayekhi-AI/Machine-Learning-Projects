import torch
import numpy as np
from core.entities.ocr import OCRResult
from core.ocr.data.transforms import preprocess_image

class OCRPredictor:
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
    probs = torch.softmax(logits, dim=2)
    max_probs, max_indices = probs.max(dim=2)

    texts = self.convertor.decode(logits)
    text = texts[0] if texts else ""

    confidence = []
    prev = -1
    for i, idx in enumerate(max_indices[0]):
      idx = idx.item()
      if idx != prev and idx != 0:
        confidence.append(max_probs[0][i].item())
      prev = idx

    conf = sum(confidence) / len(confidence) if confidence else 0.0


    return OCRResult(
      text=text,
      conf=conf,
    )
    

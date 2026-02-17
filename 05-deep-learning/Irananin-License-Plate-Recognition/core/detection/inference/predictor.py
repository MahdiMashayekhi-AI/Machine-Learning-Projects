import numpy as np
from core.entities.detection import DetectionResult

class Predictor:
  def __init__(self, detector):
    self._detector = detector

  
  def predict(self, image: np.ndarray, conf) -> list[DetectionResult]:
    return self._detector.predict(image, conf=conf)
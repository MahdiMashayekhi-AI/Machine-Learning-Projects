import numpy as np
from core.entities.detection import DetectionResult

class DetPredictor:
  def __init__(self, detector):
    self._detector = detector

  
  def predict(self, image: np.ndarray) -> list[DetectionResult]:
    return self._detector.predict(image)
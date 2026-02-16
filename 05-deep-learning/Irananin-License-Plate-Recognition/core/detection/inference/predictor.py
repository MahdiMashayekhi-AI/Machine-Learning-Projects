class Predictor:
  def __init__(self, detector):
    self._detector = detector

  
  def predict(self, image, conf=0.25):
    return self._detector.predict(image, conf=conf)
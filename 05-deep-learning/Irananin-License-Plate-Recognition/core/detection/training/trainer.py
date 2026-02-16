class Trainer:
  def __init__(self, detector, config):
    self._detector = detector
    self._config = config

  
  def train(self):
    self._detector.train(**self._config)
    return self._detector.validate()
  
  def export(self, format="onnx"):
    return self._detector.export(format)
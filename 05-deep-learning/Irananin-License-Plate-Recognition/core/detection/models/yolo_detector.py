from ultralytics import YOLO
from core.detection.models.base_detector import BaseDetector


class YoloDetector(BaseDetector):
  def __init__(self, model_path, device=None):
    self._model = YOLO(model_path)
    self.device = device


  def train(self, **kwargs):
    if self.device is not None:
      kwargs["device"] = self.device
    return self._model.train(**kwargs)
  

  def predict(self, image, conf):
    results = self._model.predict(image, conf=conf, device=self.device)

    outputs = []
    for result in results:
      for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        outputs.append({
          "bbox": (x1, y1, x2, y2),
          "conf": box.conf.item(),
        })
    
    return outputs
  

  def validate(self):
    return self._model.val()
  

  def export(self, format="onnx"):
    return self._model.export(format)
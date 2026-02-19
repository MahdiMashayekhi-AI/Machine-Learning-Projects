from core.entities.pipeline import PipelineResult
from core.ocr.inference.postprocess import PlateValidator

class Pipeline:
  def __init__(self, det, ocr, threshold=0.25):
    self.det = det
    self.ocr = ocr
    self.threshold = threshold
    self.validator = PlateValidator()

  def process_frame(self, frame) -> list[PipelineResult]:
    outputs: list[PipelineResult] = []
    detections = self.det.predict(frame)

    if len(detections) > 0:
      for det in detections:
        if det.conf >= self.threshold:

          cropped_img = self._cropper(frame, det.bbox)
          if cropped_img is None:
            continue

          ocr_result = self.ocr.predict(cropped_img)
          if not ocr_result.text or not self.validator.validate(ocr_result):
            continue

          outputs.append(PipelineResult(
            bbox=det.bbox,
            text=ocr_result.text,
            det_conf=det.conf,
            ocr_conf=ocr_result.conf)
          )

    outputs.sort(key=lambda x: x.det_conf, reverse=True)
    return outputs
  

  def _cropper(self, image, bbox):
    h, w = image.shape[:2]
    x1, y1, x2, y2 = bbox

    x1 = int(max(x1, 0))
    y1 = int(max(y1, 0))
    x2 = int(min(x2, w))
    y2 = int(min(y2, h))
    
    if x2 <= x1 or y2 <= y1:
      return None
    
    return image[y1:y2, x1:x2]
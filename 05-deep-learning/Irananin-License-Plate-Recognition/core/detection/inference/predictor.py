from ultralytics import YOLO


class Predictor:
  def __init__(self, model_path):
    self.model = YOLO(model_path)

  
  def predict(self, image, conf=0.25):
    outputs = []

    results = self.model.predict(image, conf=conf)
    for result in results:
      if len(result.boxes) > 0:
        for box in result.boxes:
          confidence = box.conf.item()

          x1, y1, x2, y2 = box.xyxy[0].tolist()
          x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
          coordinates = (x1, y1, x2, y2)

          cropped_image = image[y1:y2, x1:x2]

          outputs.append({
            "coordinates": coordinates,
            "confidence": confidence,
            "cropped_image": cropped_image,
          })

    return outputs

    

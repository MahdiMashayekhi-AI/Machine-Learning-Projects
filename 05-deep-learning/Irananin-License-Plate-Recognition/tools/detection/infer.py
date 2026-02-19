import cv2
import torch
from core.detection.inference.predictor import DetPredictor
from core.detection.models.yolo_detector import YoloDetector


MODEL_PATH = "./models/detection/best.pt"
SOURCE = 0
# SOURCE = "./data/detection/samples/image-1.jpg"
CONF_THRESHOLD = 0.25
SKIP_FRAMES = 2
DEVICE = ("cuda" if torch.cuda.is_available() else "cpu")


detector = YoloDetector(MODEL_PATH, CONF_THRESHOLD, DEVICE)
predictor = DetPredictor(detector)

cap = cv2.VideoCapture(SOURCE)
frame_count = 0
last_results = []
is_image = str(SOURCE).lower().endswith((".jpg", ".png", "jpeg"))

def draw_detections(image, detections):
  for det in detections:
    x1, y1, x2, y2 = det.bbox
    conf = det.conf

    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    label = f"Conf {conf * 100:.1f}%"
    cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
  
  return image


while True:
  ret, frame = cap.read()
  if not ret:
    break

  if SOURCE == 0:
    frame = cv2.flip(frame, 1)

  if is_image or (frame_count % (SKIP_FRAMES + 1) == 0):
    last_results = predictor.predict(frame)

  frame = draw_detections(frame, last_results)

  cv2.imshow("Detection", frame)
  frame_count += 1

  if is_image:
    cv2.waitKey(0)

  if cv2.waitKey(1) & 0xff == ord('q'):
    break


cap.release()
cv2.destroyAllWindows()  
import cv2
from core.detection.inference.predictor import Predictor


MODEL_PATH = "./models/detection/best.pt"
SOURCE = 0
# SOURCE = "./data/detection/samples/image-1.jpg"
CONF_THRESHOLD = 0.25
SKIP_FRAMES = 2


def draw_detections(image, detections):
  for det in detections:
    x1, y1, x2, y2 = det["coordinates"]
    conf = det["confidence"]

    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    label = f"Conf {conf * 100:.1f}%"
    cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
  
  return image


predictor = Predictor(MODEL_PATH)
cap = cv2.VideoCapture(SOURCE)

frame_count = 0
is_image = str(SOURCE).lower().endswith((".jpg", ".png", "jpeg"))
last_results = []


while True:
  ret, frame = cap.read()
  if not ret:
    break

  if SOURCE == 0:
    frame = cv2.flip(frame, 1)

  if is_image or (frame_count % (SKIP_FRAMES + 1) == 0):
    last_results = predictor.predict(frame, conf=CONF_THRESHOLD)

  frame = draw_detections(frame, last_results)

  cv2.imshow("Detection", frame)
  frame_count += 1

  if is_image:
    cv2.waitKey(0)

  if cv2.waitKey(1) & 0xff == ord('q'):
    break


cap.release()
cv2.destroyAllWindows()  
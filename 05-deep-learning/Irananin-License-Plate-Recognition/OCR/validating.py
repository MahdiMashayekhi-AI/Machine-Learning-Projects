from ultralytics import YOLO
import os

model = YOLO("runs/detect/train/weights/best.pt")

TEST_DIR = "./dataset/images/test"

def plate_text_from_result(result, names):
  items = []
  for b in result.boxes:
    cls = int(b.cls[0])
    x = float(b.xyxy[0][0])
    items.append((x, names[cls]))

  items.sort(key=lambda x: x[0])
  return "".join([x[1] for x in items])


def normalize_palte(plate):
  return plate.replace("-", "").strip()


correct = 0
total = 0

for img in os.listdir(TEST_DIR):
  gt = os.path.splitext(img)[0].split("_", 1)[1]
  r = model(f'{TEST_DIR}/{img}')[0]
  pred = plate_text_from_result(r, model.names)

  if normalize_palte(gt) == pred:
    correct += 1
  total += 1


print("OCR Accuracy:", (correct / total) * 100)
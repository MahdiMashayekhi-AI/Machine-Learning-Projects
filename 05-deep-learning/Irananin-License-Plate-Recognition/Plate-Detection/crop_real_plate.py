import os
import cv2
import torch
from ultralytics import YOLO

MODEL_PATH = "./runs/detect/lpd_model_v2/weights/best.pt"
IMAGES_PATH = "./dataset/valid/images/"
OUTPUT_PATH = "./outputs/real_plates/valid/"

os.makedirs(OUTPUT_PATH, exist_ok=True)

model = YOLO(MODEL_PATH)


def crop_plate(image_path, output_path):

  img = cv2.imread(image_path)
  if img is None:
    print(f"Failed to load image: {image_path}")
    return

  results = model(image_path)

  for i, result in enumerate(results):
    boxes = result.boxes.xyxy.cpu().numpy()
    for j, box in enumerate(boxes):
      x1, y1, x2, y2 = box.astype(int)
      plate = img[y1:y2, x1:x2]

      if plate.size == 0:
        continue

      basename = os.path.splitext(os.path.basename(image_path))[0]
      savename = f"{basename}_plate_{j}.jpg"
      savepath = os.path.join(output_path, savename)

      cv2.imwrite(savepath, plate)

if __name__ == "__main__":
    for image in os.listdir(IMAGES_PATH):
        if image.lower().endswith(('.png', '.jpg', '.jpeg')):
            full_path = os.path.join(IMAGES_PATH, image)
            crop_plate(full_path, OUTPUT_PATH)
            print(f"Processed: {image}")

    print("\nDone.")
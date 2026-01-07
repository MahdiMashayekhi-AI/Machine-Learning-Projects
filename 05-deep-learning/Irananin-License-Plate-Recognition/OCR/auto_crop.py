from ultralytics import YOLO
import cv2
import os

model = YOLO("runs/detect/train/weights/best.pt")

input_folder = "dataset/images/train"
output_root = "cropped_characters"
os.makedirs(output_root, exist_ok=True)

valid_ext = (".jpg", ".jpeg", ".png")

for img_name in os.listdir(input_folder):
    img_path = os.path.join(input_folder, img_name)

    if os.path.isdir(img_path):
        continue

    if not img_name.lower().endswith(valid_ext):
        continue

    img = cv2.imread(img_path)
    if img is None:
        print("Cant read:", img_path)
        continue

    results = model(img)[0]
    h, w = img.shape[:2]

    for i, box in enumerate(results.boxes):
        cls = int(box.cls)

        class_folder = os.path.join(output_root, str(model.names[cls]))
        os.makedirs(class_folder, exist_ok=True)

        pad = 8

        x1, y1, x2, y2 = box.xyxy[0].tolist()

        x1, y1 = max(0, int(x1 - pad)), max(0, int(y1 - pad))
        x2, y2 = min(w, int(x2 + pad)), min(h, int(y2 + pad))

        if x2 <= x1 or y2 <= y1:
            continue

        crop = img[y1:y2, x1:x2]

        save_path = os.path.join(class_folder, f"{img_name}_idx{i}.jpg")
        cv2.imwrite(save_path, crop)

    print(f"Done: {img_name}")

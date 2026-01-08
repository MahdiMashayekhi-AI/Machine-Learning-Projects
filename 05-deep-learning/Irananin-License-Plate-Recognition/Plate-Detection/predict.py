import cv2
import os
from ultralytics import YOLO
from paddleocr import PaddleOCR


model = YOLO('runs/detect/lpd_model_v1/weights/best.pt')
ocr = PaddleOCR(use_angle_cls=True, lang='fa') 
source = 'lpd_dataset/images/test/242_jpg.rf.7146d8a520c07948c67ca0aa98fa817f.jpg'

results = model.predict(source, conf=0.25, verbose=False)

img = cv2.imread(source)
h, w, _ = img.shape

print("-" * 50)
for result in results:
    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        
        pad = 10 
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(w, x2 + pad)
        y2 = min(h, y2 + pad)
        
        cropped_plate = img[y1:y2, x1:x2]
        temp_path = "temp_plate_crop.jpg"
        cv2.imwrite(temp_path, cropped_plate)
        
        try:
            ocr_result = ocr.predict(temp_path)
            
            if ocr_result:
                text = ocr_result[0]['rec_texts'][0]
                score = ocr_result[0]['rec_scores'][0]
                
                print(f"Plate Found: {text} (Score: {score:.2f})")
                
                with open('results.txt', 'a', encoding='utf-8') as f:
                    f.write(f"{text}\n")
            else:
                print("YOLO found a box, but OCR read nothing.")
                
        except Exception as e:
            print(f"Error: {e}")

        if os.path.exists(temp_path):
            os.remove(temp_path)

print("-" * 50)
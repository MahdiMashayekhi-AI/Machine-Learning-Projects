import torch
import cv2
from ultralytics import YOLO
from configs.config_loader import cfg

MODEL_PATH = cfg['detection']['model_path']
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = YOLO(MODEL_PATH)

results = model.track("video-plates.mp4", device=DEVICE, show=False, stream=True)

for r in results:
    frame = r.plot()

    frame = cv2.resize(frame, (800, 600))

    cv2.imshow("Video", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cv2.destroyAllWindows()
import cv2
import json
import torch
from configs.config_loader import cfg
from core.pipeline import Pipeline
from core.detection.inference.predictor import DetPredictor
from core.detection.models.yolo_detector import YoloDetector
from core.ocr.models.crnn import CRNN
from core.ocr.data.label_converter import LabelConverter
from core.ocr.inference.predictor import OCRPredictor
from core.visualization.drawer import Drawer

IMAGE_PATH = "./data/detection/samples/2.jpg"
DET_MODEL_PATH = cfg["detection"]["model_path"]
OCR_MODEL_PATH = cfg["ocr"]["model_path"]
CLASSES_PATH = cfg["ocr"]["classes_path"]
DEVICE = ("cuda" if torch.cuda.is_available() else "cpu")

with open(CLASSES_PATH, "r", encoding="utf-8") as f:
  classes = json.load(f)["classes"]

# Detection
detector = YoloDetector(DET_MODEL_PATH, DEVICE)
det = DetPredictor(detector)

# OCR
model = CRNN(1, len(classes)+1).to(DEVICE)
model.load_state_dict(torch.load(OCR_MODEL_PATH, map_location=DEVICE)["model"])

convertor = LabelConverter(classes)
ocr = OCRPredictor(model, convertor, DEVICE)


def run_image(path, pipeline, drawer):
    frame = cv2.imread(path)

    results = pipeline.process_frame(frame)

    frame = drawer.draw(frame, results)

    cv2.imshow("Result", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def run_video(path, pipeline, drawer):
    cap = cv2.VideoCapture(path)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = pipeline.process_frame(frame)
        frame = drawer.draw(frame, results)

        cv2.imshow("Video", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def run_webcam(pipeline, drawer):
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break


        results = pipeline.process_frame(frame)
        frame = drawer.draw(frame, results)

        cv2.imshow("Webcam", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    pipeline = Pipeline(detector, ocr, threshold=0.25)
    drawer = Drawer()

    mode = "image"  # image | video | webcam

    if mode == "image":
        run_image(IMAGE_PATH, pipeline, drawer)

    elif mode == "video":
        run_video("test.mp4", pipeline, drawer)

    elif mode == "webcam":
        run_webcam(pipeline, drawer)
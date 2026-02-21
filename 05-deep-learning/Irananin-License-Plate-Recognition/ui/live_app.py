import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)


import cv2
import json
import torch
import streamlit as st
from configs.config_loader import cfg
from core.pipeline import Pipeline
from core.detection.inference.predictor import DetPredictor
from core.detection.models.yolo_detector import YoloDetector
from core.ocr.models.crnn import CRNN
from core.ocr.data.label_converter import LabelConverter
from core.ocr.inference.predictor import OCRPredictor
from core.utils.mapping import map_plate

import threading

class VideoStream:
    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 1) 
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False

    def start(self):
        threading.Thread(target=self.update, args=()).start()
        return self

    def update(self):
        while True:
            if self.stopped:
                return
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True
        self.stream.release()


@st.cache_resource
def load_pipeline():
  DET_MODEL_PATH = cfg['detection']['model_path']
  OCR_MODEL_PATH = cfg['ocr']['model_path']
  CLASSES_PATH = cfg['ocr']['classes_path']
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
  predictor = OCRPredictor(model, convertor, DEVICE)

  return Pipeline(det, predictor)


st.title("🎥 License Plate Detection (Real-time)")
run = st.checkbox("Show video", value=False, key="show_video")

FRAME_WINDOW = st.image([])

camera_source = st.text_input("IP address or camera index", value="0")
source = int(camera_source) if camera_source.isdigit() else camera_source


if run:
    if "http" in str(camera_source):
        v_stream = VideoStream(source).start()
    else:
        source = int(camera_source) if camera_source.isdigit() else camera_source
        v_stream = VideoStream(source).start()
    
    while run:
        frame = v_stream.read()

        results = load_pipeline().process_frame(frame)

        for result in results:
            x1, y1, x2, y2 = result.bbox
            label = map_plate(result.text)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        frame_show = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(frame_show)
    
    run = False
    v_stream.release()
    st.info("Camera stopped and released.")
else:
    st.info("Camera is currently OFF.")
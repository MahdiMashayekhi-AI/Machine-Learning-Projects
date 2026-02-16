import torch
from core.detection.models.yolo_detector import YoloDetector
from core.detection.training.trainer import Trainer
from configs.det import MODEL_PATH, TRAIN_CONFIG

DEVICE = ("cuda" if torch.cuda.is_available() else "cpu")


def main():
  detector = YoloDetector(MODEL_PATH, device=DEVICE)
  trainer = Trainer(detector, TRAIN_CONFIG)
  trainer.train()
  trainer.export("onnx")


if __name__ == "__main__": 
  main()
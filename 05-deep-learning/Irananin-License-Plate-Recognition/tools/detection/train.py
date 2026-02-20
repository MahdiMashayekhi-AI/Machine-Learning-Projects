import torch
from configs.config_loader import cfg
from core.detection.training.trainer import Trainer
from core.detection.models.yolo_detector import YoloDetector

MODEL_PATH = cfg["detection"]["model_path"]
TRAIN_CONFIG = cfg["detection"]["training"]
DEVICE = ("cuda" if torch.cuda.is_available() else "cpu")


def main():
  detector = YoloDetector(MODEL_PATH, device=DEVICE)
  trainer = Trainer(detector, TRAIN_CONFIG)
  trainer.train()
  trainer.export("onnx")


if __name__ == "__main__": 
  main()
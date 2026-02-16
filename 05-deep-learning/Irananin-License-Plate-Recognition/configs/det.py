MODEL_PATH = "./models/detection/backbones/yolo11n.pt"
CONF_THRESHOLD = 0.25


TRAIN_CONFIG = {
  "data" : "./data/detection/raw/data.yaml",
  "epochs" : 100,
  "batch" : 32,
  "imgsz" : 640,
  "save" : True,
}

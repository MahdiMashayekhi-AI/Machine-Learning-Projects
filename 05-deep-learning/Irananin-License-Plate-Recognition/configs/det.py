MODEL_PATH = "./models/detection/backbones/yolo11n.pt"
CONF_THRESHOLD = 0.25


TRAIN_CONFIG = {
  "data" : "./data/detection/raw/data.yaml",
  "epochs" : 50,
  "batch" : 16,
  "imgsz" : 640,
  "save" : True,
  "workers" : 0,
  "patience" : 10,
  "pretrained" : True,
  "optimizer" : 'auto',
  "verbose" : True,
}

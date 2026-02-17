import torch

PAD_CHAR = "_"
MAX_LABEL_LEN = 8
IMAGE_SIZE = (200, 50)
BATCH_SIZE = 64
EPOCHS = 25
LR = 1e-3

EXPERIMENT_ROOT = "./experiments/ocr"
DATASET_DIR = "./data/ocr/raw"
CLASSES_PATH = "./configs/classes.json"

LOGGER_DIR = "./logs"
LOGGER_FILE = "result.csv"

CHECKPOINT_DIR = "./checkpoints"
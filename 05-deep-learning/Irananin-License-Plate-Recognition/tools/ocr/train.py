import os
import json
import torch
import torch.optim as optim
from configs.config_loader import cfg
from core.ocr.models.crnn import CRNN
from core.ocr.training.trainer import Trainer
from core.ocr.training.train import train
from core.ocr.training.validate import validate
from core.ocr.data.label_converter import LabelConvertor
from core.ocr.data.dataset import PlateDataset
from core.ocr.losses.ctc_loss import build_ctc_loss
from core.ocr.utils.weights import weights_init
from core.ocr.utils.logger import Logger
from core.ocr.utils.checkpoint import Checkpoint
from experiments.manager import create_experiment
from torch.utils.data import DataLoader


CLASSES_PATH = cfg['ocr']['classes_path']
LOGGER_DIR = cfg['ocr']['logs']
DATASET_DIR = cfg['dirs']['dataset']
EXPERIMENTS_DIR = cfg['dirs']['experiments']
CHECKPOINT_DIR = cfg['ocr']['checkpoints']
BATCH_SIZE = cfg['ocr']['batch_size']
EPOCHS = cfg['ocr']['epoch']
LR = cfg['ocr']['lr']
RESUME = cfg['ocr']['resume_training']
DEVICE = ("cuda" if torch.cuda.is_available() else "cpu")

exp_path = create_experiment(EXPERIMENTS_DIR)
LOGGER_DIR = os.path.join(exp_path, LOGGER_DIR)
CHECKPOINT_DIR = os.path.join(exp_path, CHECKPOINT_DIR)



with open(CLASSES_PATH, "r", encoding="utf-8") as f:
  classes = json.load(f)["classes"]

model = CRNN(1, len(classes)+1).to(DEVICE)
model.apply(weights_init)
  
convertor = LabelConvertor(classes)

train_ds = PlateDataset(DATASET_DIR, "train")
test_ds = PlateDataset(DATASET_DIR, "test")

train_loader = DataLoader(train_ds, BATCH_SIZE, True) 
test_loader = DataLoader(test_ds, BATCH_SIZE, False)

criterion = build_ctc_loss(blank=0)

optimizer = optim.Adam(model.parameters(), LR)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", factor=0.5, patience=5, min_lr=1e-5)

logger = Logger(LOGGER_DIR, print_console=True)

checkpoint = Checkpoint(CHECKPOINT_DIR)

trainer = Trainer(
  model, train, validate, convertor,
  train_loader, test_loader, criterion,
  optimizer, scheduler, logger, checkpoint, DEVICE,
)

if RESUME:
  trainer.resume(load_best=False)

trainer.fit(EPOCHS)
import os
import json
import torch
import torch.optim as optim
from configs import ocr
from core.ocr.models.crnn import CRNN
from core.ocr.training.trainer import Trainer
from core.ocr.training.train import train
from core.ocr.training.validate import validate
from core.ocr.data.lablel_convertor import LabelConvertor
from core.ocr.data.dataset import PlateDataset
from core.ocr.losses.ctc_loss import build_ctc_loss
from core.ocr.utils.weights import weights_init
from core.ocr.utils.logger import Logger
from core.ocr.utils.checkpoint import Checkpoint
from experiments.manager import create_experiment
from torch.utils.data import DataLoader


CLASSES_PATH = ocr.CLASSES_PATH
DATASET_DIR = ocr.DATASET_DIR
BATCH_SIZE = ocr.BATCH_SIZE
EPOCHS = ocr.EPOCHS
LR = ocr.LR
DEVICE = ocr.DEVICE

exp_path = create_experiment("./experiments/ocr")
LOGGER_DIR = os.path.join(exp_path, ocr.LOGEER_DIR)
CHECKPOINT_DIR = os.path.join(exp_path, ocr.CHECKPOINT_DIR)

RESUME = False


with open(CLASSES_PATH, "r", encoding="utf-8") as f:
  classes = json.load(f)["classes"]

model = CRNN(1, len(classes)+1).to(DEVICE)
model.apply(weights_init)
  
convertor = LabelConvertor(classes)

train_ds = PlateDataset(DATASET_DIR, "train")
test_ds = PlateDataset(DATASET_DIR, "test")

# small_train_ds = torch.utils.data.Subset(train_ds, range(32)) # For test
# small_test_ds = torch.utils.data.Subset(test_ds, range(32)) # For test

train_loader = DataLoader(train_ds, BATCH_SIZE, True) # For test
test_loader = DataLoader(test_ds, BATCH_SIZE, False) # For test

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
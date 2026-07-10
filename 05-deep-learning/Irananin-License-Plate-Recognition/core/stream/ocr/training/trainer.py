from time import time
from datetime import datetime
from core.ocr.utils import metrics


class Trainer:
  def __init__(self, model, train, validate, convertor, train_loader, val_loader,
        criterion, optimizer, scheduler, logger, checkpoint, device):
    self.model = model
    self.train = train
    self.validate = validate
    self.convertor = convertor
    self.train_loader = train_loader
    self.val_loader = val_loader
    self.criterion = criterion
    self.optimizer = optimizer
    self.scheduler = scheduler
    self.logger = logger
    self.checkpoint = checkpoint
    self.device = device

    self.best_loss = float("inf")
    self.start_epoch = 0


  def resume(self, load_best=False):
    epoch, best_loss = self.checkpoint.load(
      self.model,
      self.optimizer,
      self.scheduler, 
      load_best, 
      self.device)

    self.start_epoch = epoch + 1
    self.best_loss = best_loss

    print(f"Resumed from epoch {epoch}")
  
  def fit(self, epochs):
    for epoch in range(self.start_epoch, epochs):

      start_time = time()

      train_loss = self.train(
        self.model, self.train_loader, self.convertor,
        self.optimizer, self.criterion, self.device, epoch
      )

      val_output = self.validate(
        self.model, self.val_loader, self.convertor,
        self.criterion, self.device
      )

      if len(val_output["preds"]) > 0:
        preds = val_output["preds"][0]

      print(f"Sample -> GT: {val_output['targets'][0]} | PR: {preds}")

      self.scheduler.step(val_output["loss"])

      if val_output["loss"] < self.best_loss:
        self.best_loss = val_output["loss"]
        self.checkpoint.save_best(self.model, epoch, self.optimizer, self.scheduler, val_output["loss"], self.best_loss)

      self.checkpoint.save_last(self.model, epoch, self.optimizer, self.scheduler, self.best_loss)

      cer = metrics.cer(val_output["preds"], val_output["targets"])
      wer = metrics.wer(val_output["preds"], val_output["targets"])
      acc = metrics.sequence_accuracy(val_output["preds"], val_output["targets"])

      epoch_time = time() - start_time

      self.logger.log(**{
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "val_loss": val_output["loss"],
        "cer": cer,
        "wer": wer,
        "seq_acc": acc,
        "lr": self.optimizer.param_groups[0]['lr'],
        "epoch_time": epoch_time,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M")
      })
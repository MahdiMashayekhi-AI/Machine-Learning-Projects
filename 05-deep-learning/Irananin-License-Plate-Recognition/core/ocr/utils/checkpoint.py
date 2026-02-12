import os
import torch


class Checkpoint:
  def __init__(self, dir_path):
    self.dir_path = dir_path
    os.makedirs(self.dir_path, exist_ok=True)

    self.best_path = os.path.join(self.dir_path, "best.pth")
    self.last_path = os.path.join(self.dir_path, "last.pth")


  def save_last(self, model, epoch, optimizer, scheduler, best_loss):
    torch.save({
      "epoch": epoch,
      "model": model.state_dict(),
      "optimizer": optimizer.state_dict(),
      "scheuler": scheduler,
      "best_loss": best_loss,
    }, self.last_path)


  def save_best(self, model, epoch, optimizer, scheduler, metric, best_loss):
    torch.save({
      "epoch": epoch,
      "model": model.state_dict(),
      "optimizer": optimizer.state_dict(),
      "scheuler": scheduler,
      "metric": metric,
      "best_loss": best_loss,
    }, self.best_path)


  def load(self, model, optimizer=None, scheduler=None, load_best=None, device="cpu"):
    path = self.best_path if load_best else self.last_path

    if not os.path.exists(path):
      raise FileNotFoundError(f"No checkpoint found at {path}")
    
    checkpoint = torch.load(path, map_location=device)

    model.load_state_dict(checkpoint["model"])

    if optimizer:
      optimizer.load_state_dict(checkpoint["optimizer"])

    if scheduler:
      scheduler.load_state_dict(checkpoint["scheduler"])

    epoch = checkpoint["epoch"]
    best_loss = checkpoint["best_loss"]

    return epoch, best_loss
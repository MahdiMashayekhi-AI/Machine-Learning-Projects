import os
import csv


class Logger:
  def __init__(self, dir_path, print_console=False):
    self.dir_path = dir_path
    self.path = os.path.join(dir_path, "results.csv")
    self.print_console = print_console

    self.header = [
        "epoch",
        "train_loss",
        "val_loss",
        "cer",
        "wer",
        "seq_acc",
        "lr",
        "epoch_time",
        "timestamp"
    ]

    os.makedirs(self.dir_path, exist_ok=True)

    if not os.path.exists(self.path):
      with open(self.path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(self.header)


  def log(self, **metrics):
    row = [metrics.get(k) for k in self.header]

    with open(self.path, "a", encoding="utf-8", newline="") as f:
      writer = csv.writer(f)
      writer.writerow(row)


    if self.print_console:
        self._print(metrics)

  def _print(self, metrics):
      msg = (
          f"[Epoch {metrics.get('epoch')}] "
          f"train={metrics.get('train_loss'):.4f} | "
          f"val={metrics.get('val_loss'):.4f} | "
          f"CER={metrics.get('cer'):.4f} | "
          f"Acc={metrics.get('seq_acc'):.4f} | "
          f"lr={metrics.get('lr'):.6f}\n"
      )
      print(msg)
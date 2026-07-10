import torch.nn as nn

def build_ctc_loss(blank=0):
  return nn.CTCLoss(blank=blank, zero_infinity=True)
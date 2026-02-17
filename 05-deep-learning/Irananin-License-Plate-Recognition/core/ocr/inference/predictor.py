import torch

class Predictor:
  def __init__(self, model, convertor, device):
    self.model = model
    self.convertor = convertor
    self.device = device
    self.model.eval()


  @torch.no_grad()
  def predict(self, images):
    images = images.to(self.device)

    logits = self.model(images)
    
    texts = self.convertor.decode(logits)
    return texts
    

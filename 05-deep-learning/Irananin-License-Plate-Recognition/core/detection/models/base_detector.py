import numpy as np

class BaseDetector: 
  
  def train(self, **kwargs):
    raise NotImplementedError
  

  def predict(self, image: np.ndarray, conf):
    raise NotImplementedError
  

  def validate(self):
    raise NotImplementedError
  

  def export(self, format):
    raise NotImplementedError
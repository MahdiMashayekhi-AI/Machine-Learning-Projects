import numpy as np
from abc import ABC, abstractmethod

class BaseDetector(ABC): 
  
  @abstractmethod
  def train(self, **kwargs):
    raise NotImplementedError
  
  @abstractmethod
  def predict(self, image: np.ndarray, conf):
    raise NotImplementedError
  
  @abstractmethod
  def validate(self):
    raise NotImplementedError
  
  @abstractmethod
  def export(self, format):
    raise NotImplementedError
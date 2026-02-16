class BaseDetector: 
  
  def train(self, **kwargs):
    raise NotImplementedError
  

  def predict(self, image, conf):
    raise NotImplementedError
  

  def validate(self):
    raise NotImplementedError
  

  def export(self, format):
    raise NotImplementedError
  

  def load(self, path):
    raise NotImplementedError
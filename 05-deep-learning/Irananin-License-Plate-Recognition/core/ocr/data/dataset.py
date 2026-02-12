import os
from torch.utils.data import Dataset
from core.ocr.data.transforms import preprocess_image
from configs import ocr

PAD_CHAR = ocr.PAD_CHAR
MAX_LABEL_LEN = ocr.MAX_LABEL_LEN


class PlateDataset(Dataset):
  def __init__(self, root_dir, mode="train"):
    self.samples = []
    self.root_dir = root_dir
    self.label_path = os.path.join(root_dir, f"{mode}_labels.txt")

    with open(self.label_path, "r", encoding="utf-8") as f:
      for line in f:
        img, text = line.strip().split(maxsplit=1)
        if len(text) > MAX_LABEL_LEN:
          continue
        self.samples.append((img, text))

  
  def __len__(self):
    return len(self.samples)
  

  def __getitem__(self, index):
    img_path_rel, text = self.samples[index]
    img_path = os.path.join(self.root_dir, img_path_rel)
    image = preprocess_image(img_path)
    return image, text
import torch
from configs import ocr

PAD_CHAR = ocr.PAD_CHAR
MAX_LABEL_LEN = ocr.MAX_LABEL_LEN


class LabelConverter:
  def __init__(self, classes_list, pad_char=PAD_CHAR, max_len= MAX_LABEL_LEN):
    self.classes_list = classes_list
    self.pad_char = pad_char
    self.max_len = max_len

    self.cls2id = {cls : id+1 for id, cls in enumerate(classes_list)}
    self.id2cls = {id+1 : cls for id, cls in enumerate(classes_list)}


  def encode(self, text_list):
    # 28f65244 -> [2, 8, 15, 6, 5, 2, 4, 4]
    
    lengths = []
    targets = []

    for text in text_list:
      # remove padding if exists
      text = text.replace(self.pad_char, "")
      lengths.append(len(text))

      for ch in text:
        if ch not in self.cls2id:
          raise ValueError(f"Unknown character: {ch}")
        targets.append(self.cls2id[ch])

    return torch.IntTensor(targets), torch.IntTensor(lengths)

  def decode(self, preds):
    # preds: [b, t, c]
    preds = torch.argmax(preds, dim=2)
    texts = []

    for seq in preds:
      prev = 0
      res = ""
      for k in seq:
        k = int(k)
        if k != prev and k != 0:
          res += self.id2cls[k]
        prev = k
      texts.append(res)

    return texts

import torch

class LabelConvertor:
    def __init__(self, class_list, pad_char="_", max_len=8):
        self.class_list = class_list
        self.pad_char = pad_char
        self.max_len = max_len

        self.cls2id = {cls: i+1 for i, cls in enumerate(class_list)}
        self.id2cls = {i+1: cls for i, cls in enumerate(class_list)}

    def encode(self, text_list):
        """
            "27j23647" -> [3, 8, 15, 3, 4, 7, 5, 5]
        """
        targets = []
        lengths = []

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
        """
            [3, 8, 15, 3, 4, 7, 5, 5] -> "27j23647"
        """
        
        preds = preds.argmax(2).transpose(1, 0)
        texts = []

        for seq in preds:
            prev = 0
            res = ""
            for k in seq:
                k = int(k)
                if k != 0 and k != prev:
                    res += self.id2cls[k]
                prev = k
            texts.append(res)

        return texts

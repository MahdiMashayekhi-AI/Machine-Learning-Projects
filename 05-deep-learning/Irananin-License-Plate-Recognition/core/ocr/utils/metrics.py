import editdistance


def cer(preds, targets):
  # Character Error Rate
  total_dist = 0
  total_chars = 0

  for p, t in zip(preds, targets):
    total_dist += editdistance.eval(p, t)
    total_chars += len(t)

  return total_dist / max(1, total_chars)

  
def wer(preds, targets):
  # Word Error Rate
  errors = sum(p != t for p, t in zip(preds, targets))
  return errors / len(targets)
  

def sequence_accuracy(preds, targets):
  corrects = sum(p == t for p, t in zip(preds, targets))
  return corrects / len(targets)
  

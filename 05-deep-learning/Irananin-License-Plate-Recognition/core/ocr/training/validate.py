import torch
from tqdm import tqdm


def validate(model, dataloader, convertor, criterion, device):
  model.eval()
  total_loss = 0.0

  all_preds = []
  all_targets = []

  with torch.no_grad():
    pbar = tqdm(dataloader, desc=f"Validating: ")
    for images, texts in pbar:
      images = images.to(device)
      targets, lengthes = convertor.encode(texts)
      targets, lengthes = targets.to(device), lengthes.to(device)

      logits = model(images)
      log_probs = logits.log_softmax(2).permute(1, 0, 2)

      input_lengths = torch.full(
        size=(logits.size(0),),
        fill_value=log_probs.size(0),
        dtype=torch.long,
        device=device
      )

      loss = criterion(log_probs, targets, input_lengths, lengthes)
      total_loss += loss.item()

      preds = convertor.decode(logits)
      all_preds.extend(preds)
      all_targets.extend(texts)

    return {
        "loss": total_loss / len(dataloader),
        "preds": all_preds,
        "targets": all_targets
    }
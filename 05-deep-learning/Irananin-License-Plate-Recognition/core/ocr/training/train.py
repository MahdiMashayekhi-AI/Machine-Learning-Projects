import torch
import torch.nn as nn
from tqdm import tqdm

def train(model, dataloader, convertor, optimizer, criterion, device, epoch=None):
  model.train()
  total_loss = 0.0

  pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
  for images, texts in pbar:
    images = images.to(device)
    targets, lengths = convertor.encode(texts)
    targets, lengths = targets.to(device), lengths.to(device)

    optimizer.zero_grad()
    logits = model(images)
    log_probs = logits.log_softmax(2).permute(1, 0, 2)

    input_lengths = torch.full(
      size=(logits.size(0),),
      fill_value=log_probs.size(0),
      dtype=torch.long,
      device=device
    )

    loss = criterion(log_probs, targets, input_lengths, lengths)
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
    optimizer.step()

    total_loss += loss.item()

    pbar.set_postfix(loss=f"{loss.item():.4f}")

  return total_loss / len(dataloader)




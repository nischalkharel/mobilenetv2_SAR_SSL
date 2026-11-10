"""One training epoch and prediction over a dataset."""
import time

import torch
from torch import nn


def train_one_epoch(model, loader, optimizer, device, log_every=0, tag="train") -> dict:
    """log_every > 0 prints running loss/accuracy every log_every batches (progress only, not saved)."""
    model.train()  # dropout on, BatchNorm uses batch statistics and updates running stats
    loss_fn = nn.CrossEntropyLoss()
    total_loss, correct, seen, t0 = 0.0, 0, 0, time.time()
    for step, (images, labels) in enumerate(loader, 1):
        images, labels = images.to(device), labels.to(device)
        logits = model(images)  # (batch, num_classes)
        loss = loss_fn(logits, labels)
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite training loss: {loss.item()}")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(labels)
        correct += (logits.argmax(1) == labels).sum().item()
        seen += len(labels)
        if log_every and (step % log_every == 0 or step == len(loader)):
            print(f"    {tag} batch {step:4d}/{len(loader)}  running loss {total_loss / seen:.4f}  "
                  f"running acc {correct / seen:.4f}  {time.time() - t0:.0f}s", flush=True)
    return {"loss": total_loss / seen, "acc": correct / seen}


@torch.no_grad()
def predict(model, loader, device, log_every=0, tag="eval") -> dict:
    """Mean loss, accuracy, and per-image predictions (in loader order) in eval mode.
    log_every > 0 prints progress every log_every batches."""
    model.eval()  # dropout off, BatchNorm uses its stored running statistics
    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    total_loss, labels_all, preds_all, probs_all, t0 = 0.0, [], [], [], time.time()
    for step, (images, labels) in enumerate(loader, 1):
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        total_loss += loss_fn(logits, labels).item()
        probs = logits.softmax(1)
        labels_all.append(labels.cpu())
        preds_all.append(probs.argmax(1).cpu())
        probs_all.append(probs.cpu())
        if log_every and (step % log_every == 0 or step == len(loader)):
            print(f"    {tag} batch {step:4d}/{len(loader)}  {time.time() - t0:.0f}s", flush=True)
    labels_all, preds_all = torch.cat(labels_all), torch.cat(preds_all)
    return {
        "loss": total_loss / len(labels_all),
        "acc": (preds_all == labels_all).float().mean().item(),
        "labels": labels_all,
        "preds": preds_all,
        "probs": torch.cat(probs_all),  # (N, num_classes)
    }

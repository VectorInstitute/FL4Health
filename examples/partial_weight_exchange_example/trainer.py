from typing import Optional, Tuple

import torch
import torch.nn as nn
from transformers.modeling_outputs import SequenceClassifierOutput, SequenceClassifierOutputWithPast
from flwr.common.logger import log
from logging import INFO


def calcuate_accuracy(preds: torch.Tensor, targets: torch.Tensor) -> int:
    n_correct = int((preds == targets).sum().item())
    return n_correct


def infer(
    model: nn.Module,
    loss_function: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: str,
    max_batches: Optional[int] = None,
) -> Tuple[float, float]:
    model.to(device)
    # set model to eval mode (disable dropout etc.)
    model.eval()
    n_correct = 0
    n_total = 0
    total_loss = 0.0
    n_batches = 0
    # disable gradient calculations
    with torch.no_grad():
        for _, batch in enumerate(dataloader):
            # Used to simply consider a sample of the evaluation set if desired
            if max_batches is not None and n_batches > max_batches:
                break
            # send the batch components to proper deviceX
            ids = batch["input_ids"].to(device, dtype=torch.long)
            mask = batch["attention_mask"].to(device, dtype=torch.long)
            targets = batch["label"].to(device, dtype=torch.long)

            # forward pass
            outputs = model(input_ids=ids, attention_mask=mask)
            if type(outputs) in {SequenceClassifierOutput, SequenceClassifierOutputWithPast}:
                # For a SequenceClassifierOutput object, we want logits which are of shape (batch size, 4)
                loss = loss_function(outputs.logits, targets)
                pred_label = torch.argmax(outputs.logits, dim=1)
            else:
                # calculate loss for batch
                loss = loss_function(outputs, targets)
                pred_label = torch.argmax(outputs, dim=1)

            total_loss += loss.item()
            n_correct += calcuate_accuracy(pred_label, targets)
            n_total += targets.size(0)
            n_batches += 1
            if n_batches % 300 == 0:
                batches_to_complete = max_batches if max_batches is not None else len(dataloader)
                print(f"Completed {n_batches} of {batches_to_complete}...")

    accuracy = n_correct * 100 / n_total
    val_loss = total_loss / n_batches
    log(
        INFO,
        f"Client Validation Loss: {val_loss}," f"Client Validation Accuracy: {accuracy}",
    )
    
    # Return the accuracy over the entire validation set
    # and the average loss per batch (to match training loss calculaiton)
    return accuracy, val_loss


def train(
    model: nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    val_dataloader: torch.utils.data.DataLoader,
    loss_func: nn.Module,
    device: str,
    n_epochs: int = 1,
    n_training_steps: int = 300,
) -> float:
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.00001, weight_decay=0.001)
    # move model to the GPU (if available)
    model.to(device)
    model.train()
    total_training_steps = 0

    for epoch_number in range(n_epochs):
        if total_training_steps > n_training_steps:
            break
        print(f"Starting Epoch {epoch_number}")
        total_epoch_loss = 0.0
        total_steps_loss = 0.0
        n_correct = 0
        n_total = 0
        n_batches = 0

        for batch_number, batch in enumerate(train_dataloader):
            if total_training_steps > n_training_steps:
                break
            # send the batch components to proper device
            # ids has shape (batch size, input length = 512)
            ids = batch["input_ids"].to(device, dtype=torch.long)
            # mask has shape (batch size, input length = 512), zeros indicate padding tokens
            mask = batch["attention_mask"].to(device, dtype=torch.long)
            # targets has shape (batch size)
            targets = batch["label"].to(device, dtype=torch.long)

            # forward pass
            outputs = model(input_ids=ids, attention_mask=mask)
            if type(outputs) in {SequenceClassifierOutput, SequenceClassifierOutputWithPast}:
                # For a SequenceClassifierOutput object,
                # we want logits which are of shape (batch size, dataset_num_labels)
                loss = loss_func(outputs.logits, targets)
                pred_label = torch.argmax(outputs.logits, dim=1)
            else:
                # calculate loss for batch
                loss = loss_func(outputs, targets)
                pred_label = torch.argmax(outputs, dim=1)

            batch_loss = loss.item()
            total_steps_loss += batch_loss
            total_epoch_loss += batch_loss

            n_correct += calcuate_accuracy(pred_label, targets)
            n_total += targets.size(0)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_training_steps += 1
            n_batches += 1

        epoch_loss = total_epoch_loss / total_training_steps

        print(f"Training Loss Epoch: {epoch_loss}")
        log(
            INFO,
            f"Epoch: {epoch_number}, Client Training Loss: {epoch_loss/n_batches}," f"Client Training Accuracy: {n_correct / n_total}",
        )

    return n_correct / n_total

from logging import INFO
from typing import Tuple

import torch
import torch.nn as nn
from flwr.common.logger import log
from torch.utils.data import DataLoader
from torcheval.metrics.functional import multiclass_f1_score


def train(
    model: nn.Module, train_loader: DataLoader, loss_func: nn.Module, n_epochs: int, device: torch.device
) -> float:

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.00001, weight_decay=0.001)
    model.to(device)
    model.train()

    for epoch_number in range(1, n_epochs + 1):
        total_epoch_loss = 0.0
        n_correct = 0
        n_total = 0
        n_batches = 0

        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)

            n_correct += int((preds == targets).sum().item())
            n_total += targets.size(0)

            # loss and backward pass
            loss = loss_func(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_epoch_loss += loss
            n_batches += 1

        log(
            INFO,
            f"Epoch: {epoch_number}, Client Training Loss: {total_epoch_loss / n_batches},"
            f"Client Training Accuracy: {n_correct / n_total}",
        )

    return n_correct / n_total


def validate(
    model: nn.Module, val_loader: DataLoader, loss_func: nn.Module, device: torch.device
) -> Tuple[float, float]:
    model.to(device)
    model.eval()

    n_total = 0
    n_correct = 0
    n_batches = 0
    total_loss = 0.0

    for inputs, targets in val_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        outputs = model(inputs)
        preds = torch.argmax(outputs, dim=1)
        loss = loss_func(outputs, targets)

        total_loss += loss
        n_total += targets.size(0)
        n_correct += int((preds == targets).sum().item())
        n_batches += 1

    val_loss = total_loss / n_batches
    accuracy = n_correct / n_total
    log(
        INFO,
        f"Client Validation Loss: {val_loss}," f"Client Validation Accuracy: {accuracy}",
    )
    return val_loss, accuracy


def test(
    model: nn.Module, test_loader: DataLoader, loss_func: nn.Module, device: torch.device, num_classes: int
) -> Tuple[float, float, float]:
    model.to(device)
    model.eval()

    n_total = 0
    n_correct = 0
    n_batches = 0
    total_loss = 0.0

    all_predictions = []
    all_targets = []

    for inputs, targets in test_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        outputs = model(inputs)
        preds = torch.argmax(outputs, dim=1)

        loss = loss_func(outputs, targets)

        all_predictions.append(preds)
        all_targets.append(targets)

        total_loss += loss
        n_total += targets.size(0)
        n_correct += int((preds == targets).sum().item())
        n_batches += 1

    test_loss = total_loss / n_batches
    accuracy = n_correct / n_total
    f1_score = multiclass_f1_score(
        torch.cat(all_predictions), torch.cat(all_targets), num_classes=num_classes, average="macro"
    )

    log(
        INFO,
        f"Client Test Loss: {test_loss}," f"Client Test Accuracy: {accuracy}",
        f"Client Test f1 score: {float(f1_score)}",
    )

    return test_loss, accuracy, float(f1_score)

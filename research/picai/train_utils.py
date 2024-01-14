import torch
import torch.nn as nn

from torch.optim import Optimizer
from torch.nn.modules.loss import _Loss
from monai.data.dataloader import DataLoader


def train(model: nn.Module, train_loader: DataLoader, criterion: _Loss, optimizer: Optimizer, device: torch.device) -> None:
    model.to(device)
    criterion.to(device)
    model.train()

    for img, lbl in train_loader:
        img, lbl = img.to(device), lbl.to(device)
        train_step(model, img, lbl, criterion, optimizer)


def train_step(model: nn.Module, img: torch.Tensor, lbl: torch.Tensor, criterion: _Loss, optimizer: Optimizer) -> None:
    optimizer.zero_grad()
    pred = model(img)
    loss = criterion(pred, lbl)
    loss.backward()
    optimizer.step()


def validate(model: nn.Module, val_loader: DataLoader, criterion: _Loss, device: torch.device) -> None:
    model.to(device)
    criterion.to(device)
    model.eval()

    for img, lbl in val_loader:
        img, lbl = img.to(device), lbl.to(device)


def val_step(model: nn.Module, img: torch.Tensor, lbl: torch.Tensor, criterion: _Loss) -> None:
    with torch.no_grad():
        pred = model(img)
        _ = criterion(pred, lbl)

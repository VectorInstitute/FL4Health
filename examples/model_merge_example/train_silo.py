import argparse
from pathlib import Path
from typing import Any, Dict, Sequence

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from examples.models.cnn_model import MnistNet
from fl4health.utils.config import load_config
from fl4health.utils.load_data import load_mnist_data


def train_clients(
    models: Sequence[nn.Module],
    train_loaders: Sequence[DataLoader],
    criterions: Sequence[nn.Module],
    optimizers: Sequence[Optimizer],
    ckpt_base_path: Path,
    num_epochs: int,
    device: torch.device,
) -> None:

    assert all(len(models) == len(others) for others in [train_loaders, criterions, optimizers])

    for i, (model, loader, criterion, optimizer) in enumerate(zip(models, train_loaders, criterions, optimizers)):
        train_client(model, loader, criterion, optimizer, ckpt_base_path, num_epochs, device, i)


def train_client(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: Optimizer,
    ckpt_base_path: Path,
    num_epochs: int,
    device: torch.device,
    client_number: int,
) -> None:
    model.to(device)
    model.train()

    for _ in range(num_epochs):
        for input, target in train_loader:
            input, target = input.to(device), target.to(device)
            optimizer.zero_grad()
            preds = model(input)
            loss = criterion(preds, target)
            loss.backward()
            optimizer.step()

    model.cpu()
    ckpt_path = f"{ckpt_base_path}/{str(client_number)}.pt"
    torch.save({"model_state_dict": model.state_dict()}, ckpt_path)

    print(f"Client {str(client_number)} finished training. Checkpoint saved to: {ckpt_path}")


def main(config: Dict[str, Any], ckpt_base_path: Path, n_epochs: int, data_path: Path) -> None:
    n_clients = config["n_clients"]
    models = [MnistNet() for _ in range(n_clients)]
    loaders = [load_mnist_data(data_path, config["batch_size"])[0] for _ in range(n_clients)]
    optimizers = [torch.optim.Adam(model.parameters(), lr=0.001) for model in models]
    criterions = [torch.nn.CrossEntropyLoss() for _ in range(n_clients)]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_clients(models, loaders, criterions, optimizers, ckpt_base_path, n_epochs, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Clients Solo")
    parser.add_argument(
        "--config_path",
        action="store",
        type=str,
        help="Path to configuration file.",
        default="examples/model_merge_example/config.yaml",
    )
    parser.add_argument(
        "--data_path",
        action="store",
        type=str,
        help="Path to MNIST dataset.",
        default="examples/datasets/MNIST",
    )
    parser.add_argument(
        "--ckpt_base_path",
        action="store",
        type=str,
        help="Path to save client model checlpoints.",
        default="examples/model_merge_example",
    )
    parser.add_argument(
        "--n_epochs",
        action="store",
        type=int,
        help="Number of epochs to train each client.",
        default=20,
    )
    args = parser.parse_args()
    config = load_config(args.config_path)

    main(config, Path(args.ckpt_base_path), args.n_epochs, Path(args.data_path))

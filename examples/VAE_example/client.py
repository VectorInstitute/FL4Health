import argparse
from pathlib import Path
from typing import Tuple, Dict, Optional, Callable, Sequence

import flwr as fl
from flwr.common.logger import log
import torch
import torch.nn as nn
from flwr.common.typing import Config, Scalar
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from examples.VAE_example.models import MnistVariationalEncoder, MnistVariationalDecoder
from fl4health.clients.basic_client import BasicClient
from fl4health.utils.metrics import Metric
from fl4health.utils.losses import Losses
from fl4health.utils.sampler import DirichletLabelBasedSampler
from fl4health.utils.load_data import load_mnist_data
from fl4health.utils.sampler import LabelBasedSampler
from fl4health.model_bases.autoencoders_base import AutoEncoderType, VarioationalAE
from fl4health.tasks.autoencoder_trainer import VAETrainer


class VAEClient(VAETrainer, BasicClient):
    def __init__(self, data_path: Path, metrics: Sequence[Metric], DEVICE: torch.device) -> None:
        BasicClient.__init__(self, data_path, metrics, DEVICE)

    def get_data_loaders(self, config: Config) -> Tuple[DataLoader, DataLoader]:
        batch_size = self.narrow_config_type(config, "batch_size", int)
        sampler = DirichletLabelBasedSampler(list(range(10)), sample_percentage=0.75, beta=100)
        transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(torch.flatten)])
        train_loader, val_loader, _ = self.prepare_input(
            load_data=load_mnist_data,
            data_path=self.data_path,
            batch_size=batch_size,
            sampler=sampler,
            transform=transform,
        )

        return train_loader, val_loader

    def get_criterion(self, config: Config) -> _Loss:
        # The base_loss is the loss function used for comparing the original and generated image pixels.
        # In this example, data is in binary scale, therefore binary cross entropy is used.
        # In self.loss(), the base_loss is added to the kl divergence loss.
        base_loss = torch.nn.BCELoss(reduction="sum")
        return self.loss(config["latent_dim"], base_loss)

    def get_optimizer(self, config: Config) -> Optimizer:
        return torch.optim.Adam(self.model.parameters(), lr=0.001)

    def get_model(self, config: Config) -> nn.Module:
        encoder = MnistVariationalEncoder(input_size=784, latent_dim=config["latent_dim"])
        decoder = MnistVariationalDecoder(latent_dim=config["latent_dim"], output_size=784)
        return VarioationalAE(AutoEncoderType.VARIATIONAL_AE, encoder=encoder, decoder=decoder)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FL Client Main")
    parser.add_argument("--dataset_path", action="store", type=str, help="Path to the local dataset")
    args = parser.parse_args()

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data_path = Path(args.dataset_path)
    client = VAEClient(data_path, [], DEVICE)
    fl.client.start_numpy_client(server_address="0.0.0.0:8080", client=client)
    client.shutdown()

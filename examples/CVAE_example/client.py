import argparse
from pathlib import Path
from typing import Tuple, Dict, Optional, Callable, Sequence
from logging import INFO
from collections import Counter

import flwr as fl
from flwr.common.logger import log
import torch
import torch.nn as nn
from flwr.common.typing import Config, Scalar
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from fl4health.clients.basic_client import BasicClient
from fl4health.utils.metrics import Metric
from fl4health.utils.sampler import DirichletLabelBasedSampler
from fl4health.utils.load_data import load_mnist_data
from fl4health.utils.sampler import LabelBasedSampler
from fl4health.model_bases.autoencoders_base import AutoEncoderType, ConditionalVAE
from fl4health.tasks.autoencoder_trainer import CVAETrainer
from examples.CVAE_example.models import MnistConditionalEncoder, MnistConditionalDecoder


class CondAutoEncoderClient(CVAETrainer, BasicClient):
    def __init__(self, data_path: Path, metrics: Sequence[Metric], DEVICE: torch.device, condition: str):
        CVAETrainer.__init__(self, condition)
        BasicClient.__init__(self, data_path, metrics, DEVICE)

    def get_data_loaders(self, config: Config) -> Tuple[DataLoader, DataLoader]:
        batch_size = self.narrow_config_type(config, "batch_size", int)
        sampler = DirichletLabelBasedSampler(list(range(10)), sample_percentage=0.75, beta=100)
        sampler.set_seed(42)
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
        encoder = MnistConditionalEncoder(
            input_size=784, num_conditions=config["num_conditions"], latent_dim=config["latent_dim"]
        )
        decoder = MnistConditionalDecoder(
            latent_dim=config["latent_dim"], num_conditions=config["num_conditions"], output_size=784
        )
        return ConditionalVAE(
            AutoEncoderType.CONDITIONAL_VAE, num_conditions=config["num_conditions"], encoder=encoder, decoder=decoder
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FL Client Main")
    parser.add_argument("--dataset_path", action="store", type=str, help="Path to the local dataset")
    parser.add_argument("--condition", action="store", type=str, help="Client ID or 'label' used for CVAE")
    args = parser.parse_args()

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data_path = Path(args.dataset_path)
    client = CondAutoEncoderClient(data_path, [], DEVICE, args.condition)
    fl.client.start_numpy_client(server_address="0.0.0.0:8080", client=client)
    client.shutdown()

import argparse
from pathlib import Path
from typing import Sequence, Tuple

import flwr as fl
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from flwr.common.typing import Config
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from examples.CVAE_example.models import MnistConditionalDecoder, MnistConditionalEncoder
from fl4health.clients.basic_client import BasicClient
from fl4health.model_bases.autoencoders_base import AutoEncoderType, ConditionalVAE
from fl4health.pipeline.autoencoder_pipeline import CVAEPipeline
from fl4health.utils.load_data import load_mnist_data
from fl4health.utils.metrics import Metric
from fl4health.utils.random import set_all_random_seeds
from fl4health.utils.sampler import DirichletLabelBasedSampler


class CondAutoEncoderClient(BasicClient):
    def __init__(self, data_path: Path, metrics: Sequence[Metric], DEVICE: torch.device, condition: str) -> None:
        BasicClient.__init__(self, data_path, metrics, DEVICE)
        self.training_pipeline = CVAEPipeline(condition)

    def get_data_loaders(self, config: Config) -> Tuple[DataLoader, DataLoader]:
        batch_size = self.narrow_config_type(config, "batch_size", int)
        sampler = DirichletLabelBasedSampler(list(range(10)), sample_percentage=0.75, beta=100)
        transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(torch.flatten)])
        # To train an autoencoder-based model we need to set the data_target_transform.
        train_loader, val_loader, _ = load_mnist_data(
            data_dir=self.data_path,
            batch_size=batch_size,
            sampler=sampler,
            transform=transform,
            data_target_transform=self.training_pipeline.training_transform,
        )

        return train_loader, val_loader

    def get_criterion(self, config: Config) -> _Loss:
        # The base_loss is the loss function used for comparing the original and generated image pixels.
        # In this example, data is in binary scale, therefore binary cross entropy is used.
        base_loss = torch.nn.BCELoss(reduction="sum")
        latent_dim = self.narrow_config_type(config, "latent_dim", int)
        return self.training_pipeline.get_AE_loss(base_loss, latent_dim)

    def get_optimizer(self, config: Config) -> Optimizer:
        return torch.optim.Adam(self.model.parameters(), lr=0.001)

    def get_model(self, config: Config) -> nn.Module:
        latent_dim = self.narrow_config_type(config, "latent_dim", int)
        num_conditions = self.narrow_config_type(config, "num_conditions", int)
        encoder = MnistConditionalEncoder(input_size=784, num_conditions=num_conditions, latent_dim=latent_dim)
        decoder = MnistConditionalDecoder(latent_dim=latent_dim, num_conditions=num_conditions, output_size=784)
        return ConditionalVAE(
            AutoEncoderType.CONDITIONAL_VAE, num_conditions=num_conditions, encoder=encoder, decoder=decoder
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FL Client Main")
    parser.add_argument("--dataset_path", action="store", type=str, help="Path to the local dataset")
    parser.add_argument(
        "--condition",
        action="store",
        type=str,
        help="Specify whether to use 'label' or Clinet's ID (ex. '1', '2', etc) for CVAE",
    )
    args = parser.parse_args()
    set_all_random_seeds(args.seed)
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data_path = Path(args.dataset_path)
    client = CondAutoEncoderClient(data_path, [], DEVICE, args.condition)
    fl.client.start_numpy_client(server_address="0.0.0.0:8080", client=client)
    client.shutdown()

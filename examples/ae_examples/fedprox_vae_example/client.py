import argparse
from pathlib import Path

import flwr as fl
import torch
from flwr.common.typing import Config
from torch import nn
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchvision import transforms

from examples.ae_examples.fedprox_vae_example.models import MnistVariationalDecoder, MnistVariationalEncoder
from fl4health.clients.fed_prox_client import FedProxClient
from fl4health.model_bases.autoencoders_base import VariationalAe
from fl4health.preprocessing.autoencoders.loss import VaeLoss
from fl4health.utils.config import narrow_dict_type
from fl4health.utils.dataset_converter import AutoEncoderDatasetConverter
from fl4health.utils.load_data import ToNumpy, load_mnist_data
from fl4health.utils.sampler import DirichletLabelBasedSampler


class VaeFedProxClient(FedProxClient):
    def get_data_loaders(self, config: Config) -> tuple[DataLoader, DataLoader]:
        batch_size = narrow_dict_type(config, "batch_size", int)
        sampler = DirichletLabelBasedSampler(list(range(10)), sample_percentage=0.75, beta=100)
        # Flattening the input images to use an MLP-based variational autoencoder.
        transform = transforms.Compose([ToNumpy(), transforms.ToTensor(), transforms.Lambda(torch.flatten)])
        # Create and pass the autoencoder data converter to the data loader.
        self.autoencoder_converter = AutoEncoderDatasetConverter(condition=None)
        train_loader, val_loader, _ = load_mnist_data(
            data_dir=self.data_path,
            batch_size=batch_size,
            sampler=sampler,
            transform=transform,
            dataset_converter=self.autoencoder_converter,
        )
        return train_loader, val_loader

    def get_criterion(self, config: Config) -> _Loss:
        # The base_loss is the loss function used for comparing the original and generated image pixels.
        # We are using MSE loss to calculate the difference between the reconstructed and original images.
        base_loss = torch.nn.MSELoss(reduction="sum")
        latent_dim = narrow_dict_type(config, "latent_dim", int)
        return VaeLoss(latent_dim, base_loss)

    def get_optimizer(self, config: Config) -> Optimizer:
        return torch.optim.Adam(self.model.parameters(), lr=0.001)

    def get_model(self, config: Config) -> nn.Module:
        latent_dim = narrow_dict_type(config, "latent_dim", int)
        encoder = MnistVariationalEncoder(input_size=784, latent_dim=latent_dim)
        decoder = MnistVariationalDecoder(latent_dim=latent_dim, output_size=784)
        return VariationalAe(encoder=encoder, decoder=decoder)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FL Client Main")
    parser.add_argument("--dataset_path", action="store", type=str, help="Path to the local dataset")
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data_path = Path(args.dataset_path)
    client = VaeFedProxClient(data_path, [], device)
    fl.client.start_client(server_address="0.0.0.0:8080", client=client.to_client())
    client.shutdown()

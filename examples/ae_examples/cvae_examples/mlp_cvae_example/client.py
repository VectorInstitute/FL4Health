import argparse
from collections.abc import Sequence
from pathlib import Path

import flwr as fl
import torch
from flwr.common.typing import Config
from torch import nn
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchvision import transforms

from examples.ae_examples.cvae_examples.mlp_cvae_example.models import MnistConditionalDecoder, MnistConditionalEncoder
from fl4health.clients.basic_client import BasicClient
from fl4health.metrics.base_metrics import Metric
from fl4health.model_bases.autoencoders_base import ConditionalVae
from fl4health.preprocessing.autoencoders.loss import VaeLoss
from fl4health.utils.config import narrow_dict_type
from fl4health.utils.dataset_converter import AutoEncoderDatasetConverter
from fl4health.utils.load_data import ToNumpy, load_mnist_data
from fl4health.utils.random import set_all_random_seeds
from fl4health.utils.sampler import DirichletLabelBasedSampler


class CondAutoEncoderClient(BasicClient):
    def __init__(
        self, data_path: Path, metrics: Sequence[Metric], device: torch.device, condition: torch.Tensor
    ) -> None:
        super().__init__(data_path, metrics, device)
        # In this example, condition is based on client ID.
        self.condition_vector = condition
        # To train an autoencoder-based model we need to define a data converter that prepares the data
        # for self-supervised learning, concatenates the inputs and condition (packing) to let the data
        # fit into the training pipeline, and unpacks the input from condition for the model inference.
        # You can optionally pass a custom converter function to this class.
        self.autoencoder_converter = AutoEncoderDatasetConverter(self.condition_vector)

    def setup_client(self, config: Config) -> None:
        super().setup_client(config)
        # The unpacking function is passed to the CVAE model to unpack the input tensor to data and condition tensors.
        # Client's data is converted using autoencoder_converter in get_data_loaders.
        # This function can be initiated after data loaders are created.
        assert isinstance(self.model, ConditionalVae)
        self.model.unpack_input_condition = self.autoencoder_converter.get_unpacking_function()

    def get_data_loaders(self, config: Config) -> tuple[DataLoader, DataLoader]:
        batch_size = narrow_dict_type(config, "batch_size", int)
        sampler = DirichletLabelBasedSampler(list(range(10)), sample_percentage=0.75, beta=100)
        # ToTensor transform is used to make sure pixels stay in the range [0.0, 1.0].
        # Flattening the image data to match the input shape of the model.
        flatten_transform = transforms.Lambda(lambda x: torch.flatten(x))
        transform = transforms.Compose([ToNumpy(), transforms.ToTensor(), flatten_transform])
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
        # The input/output size is the flattened MNIST image size.
        encoder = MnistConditionalEncoder(input_size=784, latent_dim=latent_dim)
        decoder = MnistConditionalDecoder(latent_dim=latent_dim, output_size=784)
        return ConditionalVae(encoder=encoder, decoder=decoder)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FL Client Main")
    parser.add_argument("--dataset_path", action="store", type=str, help="Path to the local dataset")
    parser.add_argument(
        "--condition",
        action="store",
        type=int,
        help="Condition is the client's ID (ex. '1', '2', etc)",
    )
    parser.add_argument(
        "--num_conditions",
        action="store",
        type=int,
        help="Total number of conditions to create the condition vector. Total number of clients in this example.",
    )
    args = parser.parse_args()
    set_all_random_seeds(42)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data_path = Path(args.dataset_path)
    # Create the condition vector. This creation needs to be "consistent" across clients.
    # In this example, condition is based on client ID.
    # Client should decide how they want to create their condition vector.
    # Here we use simple one_hot_encoding but it can be any vector.
    condition_vector = torch.nn.functional.one_hot(torch.tensor(args.condition), num_classes=args.num_conditions)
    client = CondAutoEncoderClient(data_path, [], device, condition_vector)
    fl.client.start_client(server_address="0.0.0.0:8080", client=client.to_client())
    client.shutdown()

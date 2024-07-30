import argparse
import os
from functools import partial
from pathlib import Path
from typing import Sequence, Set, Tuple

import flwr as fl
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from flwr.common.typing import Config
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchvision.transforms import GaussianBlur

from examples.ae_examples.cvae_examples.conv_cvae_example.models import ConvConditionalDecoder, ConvConditionalEncoder
from fl4health.clients.basic_client import BasicClient
from fl4health.model_bases.autoencoders_base import ConditionalVae
from fl4health.preprocessing.autoencoders.loss import VaeLoss
from fl4health.utils.dataset_converter import AutoEncoderDatasetConverter
from fl4health.utils.load_data import load_cifar10_data
from fl4health.utils.metrics import Metric
from fl4health.utils.random import set_all_random_seeds
from fl4health.utils.sampler import MinorityLabelBasedSampler


def save_image_sample(
    transformed_image: torch.Tensor,
    sigma: float,
    label: int,
    dir: str = "research/cvae_experiments/conditional_oversample/",
) -> None:
    # Convert the tensor back to a PIL Image
    transformed_image_pil = transforms.ToPILImage()(transformed_image)
    # Save the transformed image to a file
    title = f"sigma{sigma}_label{label}.jpg"
    saving_path = os.path.join(dir, title)
    transformed_image_pil.save(saving_path)


def label_client_conditioning_data_converter(
    data: torch.Tensor, target: torch.Tensor, client_id: torch.Tensor, n_conditions: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    condition_vector = torch.nn.functional.one_hot(torch.tensor(client_id + target), num_classes=n_conditions)
    return torch.cat([data.view(-1), condition_vector]), data


class CondAutoEncoderClient(BasicClient):
    def __init__(
        self,
        data_path: Path,
        metrics: Sequence[Metric],
        DEVICE: torch.device,
        client_id: int,
        num_client: int,
        sigma: float,
        downsampling_ratio: float = 0.2,
        minority_labels: Set[int] = {1, 2, 3},
    ) -> None:
        super().__init__(data_path, metrics, DEVICE)
        self.downsampling_ratio = downsampling_ratio
        self.minority_labels = minority_labels
        cifar10_classes = 10
        self.client_id = client_id
        self.sigma = sigma
        self.n_conditions = num_client + cifar10_classes
        # Condition is based on the labels and client ID.
        self.autoencoder_converter = AutoEncoderDatasetConverter(
            custom_converter_function=partial(
                label_client_conditioning_data_converter, client_id=client_id, n_conditions=self.n_conditions
            ),
            condition_vector_size=self.n_conditions,
        )

    def setup_client(self, config: Config) -> None:
        super().setup_client(config)
        # The unpacking function is passed to the CVAE model to unpack the input tensor to data and condition tensors.
        # Client's data is converted using autoencoder_converter in get_data_loaders.
        # This function can be initiated after data loaders are created.
        assert isinstance(self.model, ConditionalVae)
        self.model.unpack_input_condition = self.autoencoder_converter.get_unpacking_function()

    def get_data_loaders(self, config: Config) -> Tuple[DataLoader, DataLoader]:
        batch_size = self.narrow_config_type(config, "batch_size", int)
        checkpoint_path = self.narrow_config_type(config, "checkpoint_path", str)
        # Minority sampler is used to create label distribution skew --> label imbalance.
        sampler = MinorityLabelBasedSampler(list(range(10)), self.downsampling_ratio, self.minority_labels)
        # GaussianBlur transformer from torchvision is used to create and control feature space distribution skew.
        # Sigma is the standard deviation used across both X and Y dimensions.
        # The larger the sigma value and kernel size, the more blurry the image gets.
        blur_transform = GaussianBlur(kernel_size=3, sigma=self.sigma)
        # To train an autoencoder-based model we need to set the data converter.
        train_loader, val_loader, _ = load_cifar10_data(
            data_dir=self.data_path,
            batch_size=batch_size,
            sampler=sampler,
            additional_transform=blur_transform,
            dataset_converter=self.autoencoder_converter,
        )
        iterator = iter(train_loader)
        data, labels = next(iterator)
        # Save a random sample to visualize the effect of gaussian filter.
        save_image_sample(transformed_image=data[0], sigma=self.sigma, label=labels[0], dir=checkpoint_path)
        return train_loader, val_loader

    def get_criterion(self, config: Config) -> _Loss:
        # The base_loss is the loss function used for comparing the original and generated image pixels.
        # We are using MSE loss to calculate the difference between the reconstructed and original images.
        base_loss = torch.nn.MSELoss(reduction="sum")
        latent_dim = self.narrow_config_type(config, "latent_dim", int)
        return VaeLoss(latent_dim, base_loss)

    def get_optimizer(self, config: Config) -> Optimizer:
        return torch.optim.Adam(self.model.parameters(), lr=0.0001)

    def get_model(self, config: Config) -> nn.Module:
        latent_dim = self.narrow_config_type(config, "latent_dim", int)
        encoder = ConvConditionalEncoder(latent_dim)
        decoder = ConvConditionalDecoder(latent_dim)
        return ConditionalVae(encoder=encoder, decoder=decoder)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FL Client Main")
    parser.add_argument("--dataset_path", action="store", type=str, help="Path to the local dataset")
    parser.add_argument(
        "--client_id",
        action="store",
        type=int,
        help="Client's ID (ex. '1', '2', etc) to be used as a part of the condition",
    )
    parser.add_argument(
        "--num_clients",
        action="store",
        type=int,
        help="Total number of clients to be used in creating the condition vector.",
    )
    parser.add_argument(
        "--sigma",
        action="store",
        type=float,
        help="Standard deviation to be used for gaussian blurring",
    )
    args = parser.parse_args()
    set_all_random_seeds(42)
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data_path = Path(args.dataset_path)
    client = CondAutoEncoderClient(
        data_path=data_path,
        metrics=[],
        DEVICE=DEVICE,
        client_id=args.client_id,
        num_client=args.num_clients,
        sigma=args.sigma,
    )
    fl.client.start_numpy_client(server_address="0.0.0.0:8080", client=client)
    client.shutdown()

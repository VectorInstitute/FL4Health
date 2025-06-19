import argparse
from pathlib import Path

import flwr as fl
import torch
from flwr.common import Config
from torch import nn
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from examples.models.mnist_model import MnistNet
from fl4health.clients.basic_client import BasicClient
from fl4health.metrics import Accuracy
from fl4health.preprocessing.pca_preprocessor import PcaPreprocessor
from fl4health.utils.config import narrow_dict_type
from fl4health.utils.load_data import ToNumpy, get_train_and_val_mnist_datasets
from fl4health.utils.random import set_all_random_seeds
from fl4health.utils.sampler import DirichletLabelBasedSampler


class MnistFedPcaClient(BasicClient):
    def get_data_loaders(self, config: Config) -> tuple[DataLoader, DataLoader]:
        batch_size = narrow_dict_type(config, "batch_size", int)
        pca_path = Path(narrow_dict_type(config, "pca_path", str))
        new_dimension = narrow_dict_type(config, "new_dimension", int)
        pca_preprocessor = PcaPreprocessor(pca_path)
        sampler = DirichletLabelBasedSampler(list(range(10)), sample_percentage=0.6, beta=0.75)

        # Get training and validation datasets.
        transform = transforms.Compose(
            [
                ToNumpy(),
                transforms.ToTensor(),
                transforms.Normalize((0.5), (0.5)),
            ]
        )
        train_dataset, validation_dataset = get_train_and_val_mnist_datasets(self.data_path, transform)
        train_dataset = sampler.subsample(dataset=train_dataset)
        validation_dataset = sampler.subsample(dataset=validation_dataset)

        # Apply dimensionality reduction.
        train_dataset = pca_preprocessor.reduce_dimension(new_dimension=new_dimension, dataset=train_dataset)
        validation_dataset = pca_preprocessor.reduce_dimension(new_dimension=new_dimension, dataset=validation_dataset)

        # Create dataloaders using dimension-reduced data.
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)
        return train_loader, val_loader

    def get_criterion(self, config: Config) -> _Loss:
        return torch.nn.CrossEntropyLoss()

    def get_optimizer(self, config: Config) -> Optimizer:
        return torch.optim.AdamW(self.model.parameters(), lr=0.0001)

    def get_model(self, config: Config) -> nn.Module:
        new_dimension = narrow_dict_type(config, "new_dimension", int)
        return MnistNet(input_dim=new_dimension).to(self.device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FL Client Main")
    parser.add_argument("--dataset_path", action="store", type=str, help="Path to the local dataset.")
    parser.add_argument("--seed", action="store", type=int, help="Random seed for this client.")
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data_path = Path(args.dataset_path)
    seed = args.seed

    # If the user wants to ensure that this example uses the same data as
    # the data used in the perform_pca example, then both examples
    # should use the same random seed.
    set_all_random_seeds(seed)
    client = MnistFedPcaClient(data_path, [Accuracy("accuracy")], device)
    fl.client.start_client(server_address="0.0.0.0:8080", client=client.to_client())
    client.shutdown()

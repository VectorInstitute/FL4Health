import argparse
import os
from pathlib import Path
from typing import Tuple, Optional, Dict
from logging import INFO

import flwr as fl
import torch
import torch.nn as nn
from flwr.common.typing import Config, Scalar
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from flwr.common.logger import log

from examples.vae_dim_example.mnist_model import MnistNet
from fl4health.clients.basic_client import BasicClient

from fl4health.utils.metrics import Accuracy

# Data
import torchvision.transforms as transforms
from fl4health.utils.dataset import BaseDataset, MNISTDataset
from fl4health.utils.sampler import DirichletLabelBasedSampler
from fl4health.utils.sampler import LabelBasedSampler
from torch.utils.data import Dataset

from fl4health.utils.losses import Losses, LossMeter, LossMeterType

class EncodedMNISTDataset(Dataset):
    def __init__(self, encoded_data, labels):
        self.encoded_data = encoded_data
        self.labels = labels

    def __len__(self):
        return len(self.encoded_data)

    def __getitem__(self, index):
        return self.encoded_data[index], self.labels[index]


class VAEclient(BasicClient):

    def encode_data(self, data) -> torch.Tensor:
        encoded_data = []
        data_loader = DataLoader(data, batch_size=128, shuffle=False)
        for images, _ in data_loader:
            with torch.no_grad():
                images = images.view(-1, 28*28)
                mus, _ = self.VAE_model.encoder(images)
            encoded_data.append(mus)
        encoded_data.pop()
        encoded_data = torch.stack(encoded_data)
        encoded_data = encoded_data.view(encoded_data.size(0)*encoded_data.size(1),encoded_data.size(2) )
        return encoded_data

    def load_VAE(self) -> nn.Module:
        checkpoint_path= "examples/autoencoder_example/MLP_VAE/"
        model_checkpoint_path = os.path.join(checkpoint_path, "best_VAE_model.pkl")
        autoencoder = torch.load(model_checkpoint_path)
        autoencoder.eval()
        return autoencoder

    def load_mnist_data(
        self,
        data_dir: Path,
        batch_size: int,
        sampler: Optional[LabelBasedSampler] = None,
    ) -> Tuple[DataLoader, DataLoader, Dict[str, int]]:
        """Load MNIST Dataset (training and validation set)."""
        log(INFO, f"Data directory: {str(data_dir)}")
        train_ds: BaseDataset = MNISTDataset(data_dir, train=True, transform=transforms.ToTensor())
        val_ds: BaseDataset = MNISTDataset(data_dir, train=False, transform=transforms.ToTensor())

        if sampler is not None:
            train_ds = sampler.subsample(train_ds)
            val_ds = sampler.subsample(val_ds)

        # Encode the data using the VAE model.
        encoded_train_data = self.encode_data(train_ds)
        encoded_train_ds = EncodedMNISTDataset(encoded_train_data, train_ds.targets)
        encoded_val_data = self.encode_data(val_ds)
        encoded_val_ds = EncodedMNISTDataset(encoded_val_data, val_ds.targets)
        log(INFO, "Data Encoded")

        train_loader = DataLoader(encoded_train_ds, batch_size=batch_size, shuffle=True)
        validation_loader = DataLoader(encoded_val_ds, batch_size=batch_size)
        num_examples = {"train_set": len(encoded_train_ds), "validation_set": len(encoded_val_ds)}
        return train_loader, validation_loader, num_examples
    
    def get_data_loaders(self, config: Config) -> Tuple[DataLoader, DataLoader]:
        batch_size = self.narrow_config_type(config, "batch_size", int)
        sampler = DirichletLabelBasedSampler(list(range(10)), sample_percentage=0.75, beta=100)
        train_loader, val_loader, _ = self.load_mnist_data(self.data_path, batch_size, sampler)
        return train_loader, val_loader

    def get_criterion(self, config: Config) -> _Loss:
        return torch.nn.CrossEntropyLoss()

    def get_optimizer(self, config: Config) -> Optimizer:
        return torch.optim.Adam(self.model.parameters(), lr=0.001)

    def get_model(self, config: Config) -> nn.Module:
        self.VAE_model = self.load_VAE()
        return MnistNet().to(self.device)
    

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FL Client Main")
    parser.add_argument("--dataset_path", action="store", type=str, help="Path to the local dataset")
    args = parser.parse_args()

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data_path = Path(args.dataset_path)
    client = VAEclient(data_path, [Accuracy("accuracy")], DEVICE)
    fl.client.start_numpy_client(server_address="0.0.0.0:8080", client=client)
    client.shutdown()

import argparse
from pathlib import Path
from typing import Tuple, Dict, Optional
from logging import INFO
from collections import Counter
import pickle

import flwr as fl
from flwr.common.logger import log
import torch
import torch.nn as nn
from flwr.common.typing import Config, Scalar
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from examples.autoencoder_example.ae_mnist_model import ConvAutoencoder, ConvVae
from fl4health.clients.basic_client import BasicClient
from fl4health.utils.losses import Losses
from fl4health.utils.sampler import DirichletLabelBasedSampler
from fl4health.utils.metrics import PSNR

# Data
import torchvision.transforms as transforms
from fl4health.utils.dataset import BaseDataset, MNISTDataset
from fl4health.utils.sampler import LabelBasedSampler



class AutoEncoderClient(BasicClient):
    
    def vae_loss(self, reconstruction_loss:torch.Tensor, mu:torch.Tensor, logvar:torch.Tensor) -> torch.Tensor:
        assert mu.size(0)==logvar.size(0)
        # KL Divergence loss
        kl_divergence_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        # The total VAE loss
        total_loss = reconstruction_loss + kl_divergence_loss
        return total_loss
    
    def save_data_distrubution(self, train_labels:torch.Tensor):
        # Create a dictionary to count samples per class
        class_distribution = {}
        labels_list = train_labels.detach().tolist()
        class_distribution = dict(Counter(labels_list))

        with open(f"{self.artifact_dir}distribution.pkl", 'wb') as file:
            pickle.dump(class_distribution, file)
        log(INFO, f"Data distribution saved at: {self.artifact_dir}distribution.pkl")

        pass

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

        # Save client's data distribution information in self.artifact_dir directory
        self.save_data_distrubution(train_ds.targets)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        validation_loader = DataLoader(val_ds, batch_size=batch_size)

        num_examples = {"train_set": len(train_ds), "validation_set": len(val_ds)}
        return train_loader, validation_loader, num_examples

    
    def get_data_loaders(self, config: Config) -> Tuple[DataLoader, DataLoader]:
        batch_size = self.narrow_config_type(config, "batch_size", int)
        # TODO: set the beta in the config file
        sampler = DirichletLabelBasedSampler(list(range(10)), sample_percentage=0.75, beta=100)
        train_loader, val_loader, _ = self.load_mnist_data(self.data_path, batch_size, sampler)
        return train_loader, val_loader

    def get_criterion(self, config: Config) -> _Loss:
        return nn.BCELoss(reduction='sum')

    def get_optimizer(self, config: Config) -> Optimizer:
        return torch.optim.Adam(self.model.parameters(), lr=0.01)
    
    def get_model(self, config: Config) -> nn.Module:
        variational = self.narrow_config_type(config, "variational", bool)
        if variational:
            self.variational= True
            return ConvVae().to(self.device)
        else:
            self.variational= False
            return ConvAutoencoder().to(self.device)
 
    def compute_loss(self, preds: torch.Tensor, target: torch.Tensor, mu: Optional[torch.Tensor]=None, logvar: Optional[torch.Tensor]=None) -> Losses:
        """
        Computes loss given preds and torch and the user defined criterion. Optionally includes dictionairy of
        loss components if you wish to train the total loss as well as sub losses if they exist.
        """

        if self.variational:
            assert mu is not None
            assert logvar is not None
            bce_loss = self.criterion(preds, target)
            loss = self.vae_loss(bce_loss, mu, logvar)
        else:
            loss = self.criterion(preds, target)
        losses = Losses(checkpoint=loss, backward=loss)
        return losses
   
    
    def train_step(self, input: torch.Tensor) -> Tuple[Losses, torch.Tensor]:
        """
        Given input and target, generate predictions, compute loss, optionally update metrics if they exist.
        Assumes self.model is in train model already.
        """
        # Clear gradients from optimizer if they exist
        self.optimizer.zero_grad()

        # Call user defined methods to get predictions and compute loss
        if self.variational:
            reconstruction, mu, logvar = self.predict(input)
            losses = self.compute_loss(reconstruction, input, mu, logvar)
        else:
            reconstruction = self.predict(input)
            # log(INFO, f"output size: {generated_image.shape}")
            losses = self.compute_loss(reconstruction, input)

        # Compute backward pass and update paramters with optimizer
        losses.backward.backward()
        self.optimizer.step()

        return losses, reconstruction
    
    def train_by_epochs(
        self, epochs: int, current_round: Optional[int] = None
    ) -> Tuple[Dict[str, float], Dict[str, Scalar]]:
        self.model.train()
        for local_epoch in range(epochs):
            self.train_metric_meter.clear()
            self.train_loss_meter.clear()
            for input, _ in self.train_loader:
                input = input.to(self.device)
                losses, reconstruction = self.train_step(input)
                self.train_loss_meter.update(losses)
                self.train_metric_meter.update(reconstruction, input)
                self.total_steps += 1
            metrics = self.train_metric_meter.compute()
            losses = self.train_loss_meter.compute()
            loss_dict = losses.as_dict()

            # Log results and maybe report via WANDB
            self._handle_logging(loss_dict, metrics, current_round=current_round, current_epoch=local_epoch)
            self._handle_reporting(loss_dict, metrics, current_round=current_round)

        # Return final training metrics
        return loss_dict, metrics

    def val_step(self, input: torch.Tensor) -> Tuple[Losses, torch.Tensor]:
        """
        Given input and target, compute loss, update loss and metrics
        Assumes self.model is in eval mode already.
        """

        # Get preds and compute loss
        with torch.no_grad():
            if self.variational:
                reconstruction, mu, logvar = self.predict(input)
                losses = self.compute_loss(reconstruction, input, mu, logvar)
            else:
                reconstruction = self.predict(input)
                losses = self.compute_loss(reconstruction, input)

        return losses, reconstruction 
   
   
    def validate(self) -> Tuple[float, Dict[str, Scalar]]:
        self.model.eval()
        self.val_metric_meter.clear()
        self.val_loss_meter.clear()
        with torch.no_grad():
            for input, _ in self.val_loader:
                input = input.to(self.device)
                losses, reconstruction = self.val_step(input)
                self.val_loss_meter.update(losses)
                self.val_metric_meter.update(reconstruction, input)

        # Compute losses and metrics over validation set
        losses = self.val_loss_meter.compute()
        loss_dict = losses.as_dict()
        metrics = self.val_metric_meter.compute()
        self._handle_logging(loss_dict, metrics, is_validation=True)

        # Checkpoint based on loss which is output of user defined compute_loss method
        self._maybe_checkpoint(loss_dict["checkpoint"])
        return loss_dict["checkpoint"], metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FL Client Main")
    parser.add_argument("--dataset_path", action="store", type=str, help="Path to the local dataset")
    parser.add_argument("--artifact_dir", action="store", type=str, help="Path to save client specific outputs", default="examples/autoencoder_example/distributions")
    args = parser.parse_args()

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data_path = Path(args.dataset_path)
    client = AutoEncoderClient(data_path, [PSNR("psnr")], DEVICE)
    # Add args.artifact_dir in client's attributes
    client.artifact_dir = args.artifact_dir
    fl.client.start_numpy_client(server_address="0.0.0.0:8080", client=client)
    client.shutdown()

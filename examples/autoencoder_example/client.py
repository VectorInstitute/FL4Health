import argparse
from pathlib import Path
from typing import Tuple, Dict, Optional
from logging import INFO

import flwr as fl
from flwr.common.logger import log
import torch
import torch.nn as nn
from flwr.common.typing import Config, Scalar
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from examples.autoencoder_example.ae_mnist_model import ConvAutoencoder
from fl4health.clients.basic_client import BasicClient
from fl4health.utils.load_data import load_mnist_data
from fl4health.utils.losses import Losses
from fl4health.utils.metrics import PSNR


class AutoEncoderClient(BasicClient):
    def get_data_loaders(self, config: Config) -> Tuple[DataLoader, DataLoader]:
        batch_size = self.narrow_config_type(config, "batch_size", int)
        train_loader, val_loader, _ = load_mnist_data(self.data_path, batch_size=batch_size)
        return train_loader, val_loader

    def get_criterion(self, config: Config) -> _Loss:
        return nn.MSELoss()

    def get_optimizer(self, config: Config) -> Optimizer:
        return torch.optim.Adam(self.model.parameters(), lr=0.001)

    def get_model(self, config: Config) -> nn.Module:
        return ConvAutoencoder().to(self.device)
 
    def compute_loss(self, preds: torch.Tensor, target: torch.Tensor) -> Losses:
        """
        Computes loss given preds and torch and the user defined criterion. Optionally includes dictionairy of
        loss components if you wish to train the total loss as well as sub losses if they exist.
        """
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
        generated_image = self.predict(input)
        # log(INFO, f"output size: {generated_image.shape}")
        losses = self.compute_loss(generated_image, input)

        # Compute backward pass and update paramters with optimizer
        losses.backward.backward()
        self.optimizer.step()

        return losses, generated_image
    
    def train_by_epochs(
        self, epochs: int, current_round: Optional[int] = None
    ) -> Tuple[Dict[str, float], Dict[str, Scalar]]:
        self.model.train()
        for local_epoch in range(epochs):
            self.train_metric_meter.clear()
            self.train_loss_meter.clear()
            for input, _ in self.train_loader:
                input = input.to(self.device)

                
                losses, generated_image = self.train_step(input)
                
                self.train_loss_meter.update(losses)
                self.train_metric_meter.update(generated_image, input)
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
            generated_image = self.predict(input)
            losses = self.compute_loss(generated_image, input)

        return losses, generated_image 
   
   
    def validate(self) -> Tuple[float, Dict[str, Scalar]]:
        self.model.eval()
        self.val_metric_meter.clear()
        self.val_loss_meter.clear()
        with torch.no_grad():
            for input, _ in self.val_loader:
                input = input.to(self.device)
                losses, generated_image = self.val_step(input)
                self.val_loss_meter.update(losses)
                self.val_metric_meter.update(generated_image, input)

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
    args = parser.parse_args()

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data_path = Path(args.dataset_path)
    client = AutoEncoderClient(data_path, [PSNR("psnr")], DEVICE)
    fl.client.start_numpy_client(server_address="0.0.0.0:8080", client=client)
    client.shutdown()

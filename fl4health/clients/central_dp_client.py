from fl4health.clients.basic_client import BasicClient


import pickle
from logging import DEBUG, INFO, WARN
from pathlib import Path
from random import random
from typing import Dict, Optional, Sequence, Tuple
import time
import torch
from flwr.common.logger import log
from flwr.common.typing import Config, List, NDArrays, Scalar

from fl4health.checkpointing.checkpointer import TorchCheckpointer
from fl4health.clients.basic_client import BasicClient
from fl4health.parameter_exchange.parameter_exchanger_base import ParameterExchanger
from fl4health.parameter_exchange.secure_aggregation_exchanger import SecureAggregationExchanger
from fl4health.privacy_mechanisms.index import PrivacyMechanismIndex

from fl4health.utils.losses import LossMeterType
from fl4health.utils.metrics import Metric, MetricMeterType
from fl4health.server.secure_aggregation_utils import vectorize_model

import json 
import os
import uuid 
import timeit

torch.set_default_device('cuda' if torch.cuda.is_available() else 'cpu')


class CentralDPClient(BasicClient):
    def __init__(
        self,
        *,
        data_path: Path,
        device: torch.device,
        metrics: Sequence[Metric],
        client_id: str = uuid.uuid1(),
        checkpointer: Optional[TorchCheckpointer] = None,
        loss_meter_type: LossMeterType = LossMeterType.AVERAGE,
        metric_meter_type: MetricMeterType = MetricMeterType.AVERAGE,
        task_name: str = '',
        batch_size: int = 0,
        learning_rate : float = 0,
    ) -> None:
        super().__init__(data_path, metrics, device, loss_meter_type, metric_meter_type, checkpointer)

        self.client_id = client_id
        self.task_name = task_name
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        # Since clients communicate model deltas = trained model - initial model
        # we store the initial model at /temp 
        # ("initial" refers to the start of the current FL round).
        temporary_dir = os.path.join(
            os.path.dirname(checkpointer.best_checkpoint_path),
            'temp'
        )

        if not os.path.exists(temporary_dir):
            os.makedirs(temporary_dir)

        self.temporary_model_path = os.path.join(
            temporary_dir,
            f'client_{self.client_id}_initial_model.pth'
        )

        metrics_dir = os.path.join(
            os.path.dirname(checkpointer.best_checkpoint_path),
            'metrics'
        )

        if not os.path.exists(metrics_dir):
            os.makedirs(metrics_dir)

        self.metrics_path = os.path.join(
            metrics_dir,
            f'client_{self.client_id}_metrics.json'
        )

        # will be set after model initialization
        self.model_dim = None

        with open(self.metrics_path, 'w+') as file:
            json.dump({
                'task_name': self.task_name,
                'id': self.client_id,
                'batch_size': self.batch_size
            },file)

    # Orchestrate training
    def fit(self, parameters: NDArrays, config: Config) -> Tuple[NDArrays, int, Dict[str, Scalar]]:

        local_epochs, local_steps, self.federated_round = self.process_config(config)
        log(INFO, f'Start of FL round {self.federated_round}')

        if not self.initialized:
            self.setup_client(config)
            self.model_dim = sum(param.numel() for param in self.model.state_dict().values())
            self.start_time=timeit.default_timer()

        # local model <- global model
        self.set_parameters(parameters, config)

        # store initial model for getting model delta
        initial_model_vector = vectorize_model(self.model)
        torch.save(initial_model_vector, self.temporary_model_path)
        del initial_model_vector


        if local_epochs is not None:
            log(INFO, 'Training by epochs')
            loss_dict, metrics, training_set_size = self.train_by_epochs(local_epochs, self.federated_round)
            local_steps = len(self.train_loader) * local_epochs  # total steps over training round
            self.num_train_samples = training_set_size

        elif local_steps is not None:
            log(INFO, 'Training by steps')
            loss_dict, metrics, training_set_size = self.train_by_steps(local_steps, self.federated_round)
            self.num_train_samples = training_set_size
        else:
            raise ValueError("Must specify either local_epochs or local_steps in the Config.")

        # Update after train round (Used by Scaffold and DP-Scaffold Client to update control variates)
        self.update_after_train(local_steps, loss_dict)


        with open(self.metrics_path, 'r') as file:
            metrics_to_save = json.load(file)

            metrics_to_save['model_size'] = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            metrics_to_save['learning_rate'] = self.learning_rate
            metrics_to_save['current_round'] = self.federated_round
            for key, value in metrics.items():
                if key not in metrics_to_save:
                    metrics_to_save[key] = [value]
                else:
                    metrics_to_save[key].append(value)

            for key, value in loss_dict.items():
                if key not in metrics_to_save:
                    metrics_to_save[key] = [value]
                else:
                    metrics_to_save[key].append(value)

            if 'time' not in metrics_to_save:
                metrics_to_save['time'] = [timeit.default_timer()-self.start_time]
            else:
                metrics_to_save['time'].append(timeit.default_timer()-self.start_time)
                
        with open(self.metrics_path, 'w') as file:
            json.dump(metrics_to_save, file)

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        initial_model_vector: torch.Tensor = torch.load(self.temporary_model_path).to(device=device)
        model_delta = vectorize_model(self.model) - initial_model_vector

        return (
            [model_delta.cpu().numpy()],
            self.num_train_samples,
            metrics,
        )
    
    def train_by_epochs(
        self, epochs: int, current_round: Optional[int] = None
    ) -> Tuple[Dict[str, float], Dict[str, Scalar], int]:
        """These are cutomized for Poisson subsampling"""
        self.model.train()
        local_step = 0

        datasize = 0
        for local_epoch in range(epochs):
            log(INFO, f'Consumed {datasize} datapoints by epoch {local_epoch}.')
            self.train_metric_meter_mngr.clear()
            self.train_loss_meter.clear()
            for input, target in self.train_loader:

                datasize += list(input.shape)[0]

                input, target = input.to(self.device), target.to(self.device)
                losses, preds = self.train_step(input, target)
                self.train_loss_meter.update(losses)
                self.train_metric_meter_mngr.update(preds, target)
                self.update_after_step(local_step)
                self.total_steps += 1
                local_step += 1
            metrics = self.train_metric_meter_mngr.compute()
            losses = self.train_loss_meter.compute()
            loss_dict = losses.as_dict()

            # Log results and maybe report via WANDB
            self._handle_logging(loss_dict, metrics, current_round=current_round, current_epoch=local_epoch)
            self._handle_reporting(loss_dict, metrics, current_round=current_round)

        # Return final training metrics
        return loss_dict, metrics, datasize

    def train_by_steps(
        self, steps: int, current_round: Optional[int] = None
    ) -> Tuple[Dict[str, float], Dict[str, Scalar], int]:
        """These are cutomized for Poisson subsampling"""

        # log(INFO, '===== training by steps ======')
        # for k, v in self.model.state_dict().items():
        #     log(INFO, v)

        self.model.train()

        # Pass loader to iterator so we can step through train loader
        train_iterator = iter(self.train_loader)

        self.train_loss_meter.clear()
        self.train_metric_meter_mngr.clear()

        datasize = 0
        for step in range(steps):
            # log(INFO, f'Consumed {datasize} datapoints by step {step}.')
            try:
                input, target = next(train_iterator)
            except StopIteration:
                # StopIteration is thrown if dataset ends
                # reinitialize data loader
                train_iterator = iter(self.train_loader)
                input, target = next(train_iterator)

            datasize += list(input.shape)[0]
            input, target = input.to(self.device), target.to(self.device)
            losses, preds = self.train_step(input, target)
            self.train_loss_meter.update(losses)
            self.train_metric_meter_mngr.update(preds, target)
            self.update_after_step(step)
            self.total_steps += 1

        losses = self.train_loss_meter.compute()
        loss_dict = losses.as_dict()
        # log(INFO, '==========Training losses start==========')
        # log(INFO, loss_dict)
        # log(INFO, '==========Training losses end==========')
        metrics = self.train_metric_meter_mngr.compute()

        # Log results and maybe report via WANDB
        self._handle_logging(loss_dict, metrics, current_round=current_round)
        self._handle_reporting(loss_dict, metrics, current_round=current_round)

        return loss_dict, metrics, datasize

# import torch
# import torch.nn as nn
# from flwr.common.typing import Config
# from torch.nn.modules.loss import _Loss
# from torch.optim import Optimizer
# from torch.utils.data import DataLoader

# from examples.models.cnn_model import MnistNet
# from fl4health.clients.ditto_client import DittoClient
# from fl4health.utils.config import narrow_dict_type
# from fl4health.utils.load_data import load_mnist_data
# from fl4health.utils.sampler import DirichletLabelBasedSampler

# from fl4health.utils.metrics import Accuracy
# from fl4health.reporting import JsonReporter
# from pathlib import Path

# class MnistDittoClient(DittoClient):
#     def get_data_loaders(self, config: Config) -> tuple[DataLoader, DataLoader]:
#         sample_percentage = narrow_dict_type(config, "downsampling_ratio", float)
#         sampler = DirichletLabelBasedSampler(list(range(10)), sample_percentage=sample_percentage, beta=1)
#         batch_size = narrow_dict_type(config, "batch_size", int)
#         train_loader, val_loader, _ = load_mnist_data(self.data_path, batch_size, sampler)
#         return train_loader, val_loader

#     def get_model(self, config: Config) -> nn.Module:
#         return MnistNet().to(self.device)

#     def get_optimizer(self, config: Config) -> dict[str, Optimizer]:
#         # Note that the global optimizer operates on self.global_model.parameters()
#         global_optimizer = torch.optim.AdamW(self.global_model.parameters(), lr=0.01)
#         local_optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.01)
#         return {"global": global_optimizer, "local": local_optimizer}

#     def get_criterion(self, config: Config) -> _Loss:
#         return torch.nn.CrossEntropyLoss()


# def get_client_fn(data_path: Path, device: torch.device) -> MnistDittoClient:
#     return MnistDittoClient(data_path, [Accuracy()], device, reporters=[JsonReporter()])

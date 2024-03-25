import torch
import torch.nn as nn

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
# print(f"Using {device} device")

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(2*2, 4),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)
# # print(model)

# optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

# print('----')
# # print(optimizer.param_groups[0].keys())

# # for i in optimizer.param_groups[0]['params']:
# #     print("+++")
# #     print(type(i.data), i.data)

# print(set(model.parameters()))

from research.flamby.flamby_data_utils import construct_fedisic_train_val_datasets
from torch.utils.data import DataLoader
from research.flamby_local_dp.fed_isic2019.model import ModifiedBaseline
import torch 
from itertools import chain

from flwr.common.logger import log
from logging import INFO

model = ModifiedBaseline()
optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
from opacus import PrivacyEngine

privacy_engine = PrivacyEngine()

train_dataset, validation_dataset = construct_fedisic_train_val_datasets(
    1, str('flamby_datasets/fed_isic2019')
)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, generator=torch.Generator(device='cuda' if torch.cuda.is_available() else "cpu"))


# model, optimizer, train_loader = privacy_engine.make_private(
#     module=model,
#     optimizer=optimizer,
#     data_loader=train_loader,
#     noise_multiplier=0.1,
#     max_grad_norm=1,
#     clipping="flat",
# )

# model_parameters = set(model.parameters())
# for p in chain.from_iterable(
#     [param_group["params"] for param_group in optimizer.param_groups]
# ):
#     if p not in model_parameters:
#         print('issue!')
#         raise ValueError(
#             "Module parameters are different than optimizer Parameters"
#         )

from opacus.validators import ModuleValidator
from research.flamby_local_dp.fed_ixi.model import ModifiedBaseline

model = ModifiedBaseline()

model = ModuleValidator.fix(model)
errors = ModuleValidator.validate(model, strict=False)
optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
model, optimizer, train_loader = privacy_engine.make_private(
    module=model,
    optimizer=optimizer,
    data_loader=train_loader,
    noise_multiplier=0.1,
    max_grad_norm=1,
    clipping="flat",
)

# print(errors[-5:])
print('sucess')
# log(INFO, model)
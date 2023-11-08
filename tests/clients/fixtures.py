from pathlib import Path

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from fl4health.clients.apfl_client import ApflClient
from fl4health.clients.evaluate_client import EvaluateClient
from fl4health.clients.fed_prox_client import FedProxClient
from fl4health.clients.instance_level_privacy_client import InstanceLevelPrivacyClient
from fl4health.clients.moon_client import MoonClient
from fl4health.clients.numpy_fl_client import NumpyFlClient
from fl4health.clients.scaffold_client import DPScaffoldClient, ScaffoldClient
from fl4health.model_bases.apfl_base import ApflModule
from fl4health.parameter_exchange.layer_exchanger import FixedLayerExchanger, LayerExchangerWithExclusions
from fl4health.parameter_exchange.packing_exchanger import ParameterExchangerWithPacking
from fl4health.parameter_exchange.parameter_packer import ParameterPackerFedProx, ParameterPackerWithControlVariates
from fl4health.utils.metrics import Accuracy


@pytest.fixture
def get_client(type: type, model: nn.Module) -> NumpyFlClient:
    client: NumpyFlClient
    if type == ScaffoldClient:
        client = ScaffoldClient(data_path=Path(""), metrics=[Accuracy()], device=torch.device("cpu"))
        client.learning_rate = 0.01
        model_size = len(model.state_dict()) if model else 0
        client.parameter_exchanger = ParameterExchangerWithPacking(ParameterPackerWithControlVariates(model_size))
    elif type == FedProxClient:
        client = FedProxClient(data_path=Path(""), metrics=[Accuracy()], device=torch.device("cpu"))
        client.parameter_exchanger = ParameterExchangerWithPacking(ParameterPackerFedProx())
    elif type == MoonClient:
        client = MoonClient(data_path=Path(""), metrics=[Accuracy()], device=torch.device("cpu"))
    elif type == InstanceLevelPrivacyClient:
        client = InstanceLevelPrivacyClient(data_path=Path(""), metrics=[Accuracy()], device=torch.device("cpu"))
        client.noise_multiplier = 1.0
        client.clipping_bound = 5.0
    elif type == DPScaffoldClient:
        client = DPScaffoldClient(data_path=Path(""), metrics=[Accuracy()], device=torch.device("cpu"))
        client.noise_multiplier = 1.0
        client.clipping_bound = 5.0
    elif type == ApflClient:
        client = ApflClient(data_path=Path(""), metrics=[Accuracy()], device=torch.device("cpu"))
    else:
        raise ValueError(f"{str(type)} is not a valid client type")

    client.model = model
    client.optimizer = torch.optim.SGD(client.model.parameters(), lr=0.0001)  # type: ignore
    client.train_loader = DataLoader(TensorDataset(torch.ones((1000, 28, 28, 1)), torch.ones((1000))))  # type: ignore
    client.initialized = True
    return client


@pytest.fixture
def get_evaluation_client(model: nn.Module) -> EvaluateClient:
    client = EvaluateClient(
        data_path=Path(""), metrics=[Accuracy()], device=torch.device("cpu"), model_checkpoint_path=None
    )
    client.parameter_exchanger = client.get_parameter_exchanger({})
    client.global_model = model
    client.local_model = model
    client.data_loader = DataLoader(TensorDataset(torch.ones((10, 100)), torch.ones((10), dtype=torch.long)), 5)
    client.initialized = True
    client.criterion = nn.CrossEntropyLoss()
    return client


@pytest.fixture
def get_numpy_fl_client(model: nn.Module) -> NumpyFlClient:
    client = NumpyFlClient(data_path=Path(""), device=torch.device("cpu"))
    client.parameter_exchanger = LayerExchangerWithExclusions(model, {nn.Linear})
    client.model = model
    client.initialized = True
    return client


@pytest.fixture
def get_apfl_client(type: type, model: nn.Module) -> ApflClient:
    client = ApflClient(data_path=Path(""), metrics=[Accuracy()], device=torch.device("cpu"))
    apfl_model = ApflModule(model, False)
    client.model = apfl_model
    client.parameter_exchanger = FixedLayerExchanger(apfl_model.layers_to_exchange())
    client.initialized = True
    return client

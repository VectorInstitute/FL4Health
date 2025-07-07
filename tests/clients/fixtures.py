from pathlib import Path

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from fl4health.clients.apfl_client import ApflClient
from fl4health.clients.basic_client import BasicClient
from fl4health.clients.constrained_fenda_client import ConstrainedFendaClient
from fl4health.clients.ditto_client import DittoClient
from fl4health.clients.evaluate_client import EvaluateClient
from fl4health.clients.fed_prox_client import FedProxClient
from fl4health.clients.fedper_client import FedPerClient
from fl4health.clients.fedpm_client import FedPmClient
from fl4health.clients.fedrep_client import FedRepClient
from fl4health.clients.fenda_client import FendaClient
from fl4health.clients.fenda_ditto_client import FendaDittoClient
from fl4health.clients.gpfl_client import GpflClient
from fl4health.clients.instance_level_dp_client import InstanceLevelDpClient
from fl4health.clients.moon_client import MoonClient
from fl4health.clients.mr_mtl_client import MrMtlClient
from fl4health.clients.perfcl_client import PerFclClient
from fl4health.clients.scaffold_client import DPScaffoldClient, ScaffoldClient
from fl4health.losses.fenda_loss_config import (
    ConstrainedFendaLossContainer,
    CosineSimilarityLossContainer,
    MoonContrastiveLossContainer,
    PerFclLossContainer,
)
from fl4health.metrics import Accuracy
from fl4health.model_bases.apfl_base import ApflModule
from fl4health.model_bases.fenda_base import FendaModel, FendaModelWithFeatureState
from fl4health.model_bases.gpfl_base import GpflModel
from fl4health.model_bases.masked_layers.masked_layers_utils import convert_to_masked_model
from fl4health.model_bases.parallel_split_models import ParallelSplitHeadModule
from fl4health.model_bases.perfcl_base import PerFclModel
from fl4health.parameter_exchange.fedpm_exchanger import FedPmExchanger
from fl4health.parameter_exchange.full_exchanger import FullParameterExchanger
from fl4health.parameter_exchange.layer_exchanger import FixedLayerExchanger, LayerExchangerWithExclusions
from fl4health.parameter_exchange.packing_exchanger import FullParameterExchangerWithPacking
from fl4health.parameter_exchange.parameter_packer import (
    ParameterPackerAdaptiveConstraint,
    ParameterPackerWithControlVariates,
)
from tests.test_utils.models_for_test import CompositeConvNet


@pytest.fixture
def get_client(type: type, model: nn.Module) -> BasicClient:
    if type == BasicClient:
        client = BasicClient(data_path=Path(""), metrics=[Accuracy()], device=torch.device("cpu"))
    elif type == ScaffoldClient:
        client = ScaffoldClient(data_path=Path(""), metrics=[Accuracy()], device=torch.device("cpu"))
        client.learning_rate = 0.01
        model_size = len(model.state_dict()) if model else 0
        client.parameter_exchanger = FullParameterExchangerWithPacking(ParameterPackerWithControlVariates(model_size))
    elif type == FedProxClient:
        client = FedProxClient(data_path=Path(""), metrics=[Accuracy()], device=torch.device("cpu"))
        client.parameter_exchanger = FullParameterExchangerWithPacking(ParameterPackerAdaptiveConstraint())
    elif type == DittoClient:
        client = DittoClient(data_path=Path(""), metrics=[Accuracy()], device=torch.device("cpu"))
    elif type == FedRepClient:
        client = FedRepClient(data_path=Path(""), metrics=[Accuracy()], device=torch.device("cpu"))
    elif type == FedPerClient:
        client = FedPerClient(data_path=Path(""), metrics=[Accuracy()], device=torch.device("cpu"))
    elif type == FendaDittoClient:
        client = FendaDittoClient(data_path=Path(""), metrics=[Accuracy()], device=torch.device("cpu"))
    elif type == MrMtlClient:
        client = MrMtlClient(data_path=Path(""), metrics=[Accuracy()], device=torch.device("cpu"))
    elif type == MoonClient:
        client = MoonClient(data_path=Path(""), metrics=[Accuracy()], device=torch.device("cpu"))
        client.parameter_exchanger = FullParameterExchanger()
    elif type == InstanceLevelDpClient:
        client = InstanceLevelDpClient(data_path=Path(""), metrics=[Accuracy()], device=torch.device("cpu"))
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
    client.optimizers = {"global": torch.optim.SGD(client.model.parameters(), lr=0.0001)}  # type: ignore
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
def get_basic_client(model: nn.Module) -> BasicClient:
    client = BasicClient(data_path=Path(""), metrics=[Accuracy()], device=torch.device("cpu"))
    client.model = model
    client.parameter_exchanger = LayerExchangerWithExclusions(model, {nn.Linear})
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


@pytest.fixture
def get_fenda_client(
    local_module: nn.Module, global_module: nn.Module, head_module: ParallelSplitHeadModule
) -> FendaClient:
    client = FendaClient(
        data_path=Path(""),
        metrics=[Accuracy()],
        device=torch.device("cpu"),
    )
    fenda_model = FendaModel(local_module, global_module, head_module)
    client.model = fenda_model
    client.parameter_exchanger = FixedLayerExchanger(fenda_model.layers_to_exchange())
    client.initialized = True
    return client


@pytest.fixture
def get_constrained_fenda_client(
    local_module: nn.Module, global_module: nn.Module, head_module: ParallelSplitHeadModule
) -> ConstrainedFendaClient:
    device = torch.device("cpu")
    perfcl_loss_config = PerFclLossContainer(device, 1.0, 1.0)
    contrastive_loss_config = MoonContrastiveLossContainer(device, 1.0)
    cos_sim_loss_config = CosineSimilarityLossContainer(device, 1.0)
    fenda_loss_config = ConstrainedFendaLossContainer(perfcl_loss_config, cos_sim_loss_config, contrastive_loss_config)
    client = ConstrainedFendaClient(
        data_path=Path(""),
        metrics=[Accuracy()],
        device=device,
        loss_container=fenda_loss_config,
    )
    fenda_model = FendaModelWithFeatureState(local_module, global_module, head_module, flatten_features=True)
    client.model = fenda_model
    client.parameter_exchanger = FixedLayerExchanger(fenda_model.layers_to_exchange())
    client.initialized = True
    return client


@pytest.fixture
def get_perfcl_client(
    local_module: nn.Module, global_module: nn.Module, head_module: ParallelSplitHeadModule
) -> PerFclClient:
    client = PerFclClient(
        data_path=Path(""),
        metrics=[Accuracy()],
        device=torch.device("cpu"),
        global_feature_contrastive_loss_weight=1.0,
        local_feature_contrastive_loss_weight=1.0,
    )
    perfcl_model = PerFclModel(local_module, global_module, head_module)
    client.model = perfcl_model
    client.parameter_exchanger = FixedLayerExchanger(perfcl_model.layers_to_exchange())
    client.initialized = True
    return client


@pytest.fixture
def get_fedpm_client(model: CompositeConvNet) -> FedPmClient:
    client = FedPmClient(
        data_path=Path(""),
        metrics=[Accuracy()],
        device=torch.device("cpu"),
    )
    client.model = convert_to_masked_model(model)
    client.parameter_exchanger = FedPmExchanger()
    client.initialized = True
    return client


@pytest.fixture
def get_gpfl_client(
    global_module: nn.Module, head_module: nn.Module, feature_dim: int, num_classes: int
) -> GpflClient:
    client = GpflClient(
        data_path=Path(""),
        metrics=[Accuracy()],
        device=torch.device("cpu"),
        lam=0.001,
        mu=0.001,
    )
    gpfl_model = GpflModel(
        base_module=global_module,
        head_module=head_module,
        feature_dim=feature_dim,
        num_classes=num_classes,
    )
    client.model = gpfl_model
    client.parameter_exchanger = FixedLayerExchanger(gpfl_model.layers_to_exchange())
    client.initialized = True
    return client

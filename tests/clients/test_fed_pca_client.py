from pathlib import Path

import torch
from flwr.common import Config

from fl4health.clients.fed_pca_client import FedPCAClient
from fl4health.preprocessing.pca_preprocessor import PcaPreprocessor
from fl4health.utils.random import set_all_random_seeds, unset_all_random_seeds


def test_setup_pca_client_and_save_components(tmp_path: Path) -> None:
    set_all_random_seeds(2023)

    client = FedPCAClient(data_path=Path(""), device=torch.device("cpu"), model_save_dir=tmp_path, client_name="test")
    client.initialized = True

    # Create model
    config: Config = {"low_rank": False, "full_svd": True, "rank_estimation": 6}
    client.model = client.get_model(config)
    # Fill some components to be saved
    principal_components = torch.randn((10, 20))
    singular_values = torch.randn(10, 10)
    client.model.set_principal_components(principal_components=principal_components, singular_values=singular_values)

    # Save the model in the temporary directory
    client.save_model()

    # Load it as a preprocessor module
    preprocessor = PcaPreprocessor(tmp_path / "client_test_pca.pt")

    assert torch.allclose(principal_components, preprocessor.pca_module.principal_components, atol=1e-6)
    assert torch.allclose(singular_values, preprocessor.pca_module.singular_values, atol=1e-6)

    unset_all_random_seeds()

import os
import pickle
import shutil
from collections import OrderedDict
from multiprocessing import Process
from os.path import exists, join
from typing import Dict, Optional, Tuple

import torch
from batchgenerators.utilities.file_and_folder_operations import load_json, save_json
from flwr.client import NumPyClient
from flwr.common.typing import Config, NDArrays, Scalar
from nnunetv2.experiment_planning.plan_and_preprocess_api import extract_fingerprints, preprocess_dataset
from nnunetv2.paths import nnUNet_preprocessed, nnUNet_raw, nnUNet_results
from nnunetv2.run.run_training import run_training
from nnunetv2.utilities.dataset_name_id_conversion import convert_id_to_dataset_name

# Add support for ensembling in the future


class nnUNetClient(NumPyClient):
    def __init__(
        self,
        dataset_id: int,
        device: torch.device,
        data_identifier: Optional[str],
        plans_identifier: Optional[str],
        always_preprocess: bool = False,
    ) -> None:
        self.dataset_id = dataset_id
        self.dataset_name = convert_id_to_dataset_name(dataset_id)
        self.data_identifier: Optional[str] = data_identifier
        self.plans_identifier = plans_identifier
        self.always_preprocess = always_preprocess
        self.initialized = False
        self.device = device

    def get_properties(self, config: Config) -> Dict[str, Scalar]:
        return {}  # Server does not need any properties for now

    def get_parameters(self, config: Config) -> NDArrays:
        model = torch.load(self.ckpt)
        return [val.cpu().numpy() for _, val in model["network_weights"].items()]

    def set_parameters(self, parameters: NDArrays, config: Config) -> None:
        model = torch.load(self.ckpt)
        params_dict = zip(model["network_weights"].keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        model["network_weights"] = state_dict
        torch.save(model, self.ckpt)

    def fit(self, parameters: NDArrays, config: Config) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        if not self.initialized:
            self.setup_client(config=config)

        # Check the config file
        self.check_config(config=config)

        # Set the parameters provided by server
        self.set_parameters(parameters=parameters, config=config)

        # Check to see if checkpoint exists as it will not before first round
        if not exists(self.ckpt):
            starting_ckpt = None
        else:
            starting_ckpt = self.ckpt

        # Run Training in a seperate process
        kwargs = {
            "dataset_name_or_id": str(self.dataset_id),
            "configuration": config["nnunet_config"],
            "fold": config["fold"],
            "trainer_class_name": config["nnunet_trainer"],
            "plans_identifier": self.plans_identifier,
            "pretrained_weights": starting_ckpt,
            "device": self.device,
        }
        p = Process(target=run_training, kwargs=kwargs)
        p.start()
        p.join()

        parameters = self.get_parameters(config=config)
        num_training = self.get_num_training(config=config)
        metrics: Dict[str, Scalar] = {}  # No metrics for now. To be implemented
        return parameters, num_training, metrics

    def evaluate(self, parameters: NDArrays, config: Config) -> Tuple[float, int, Dict[str, Scalar]]:
        # Set the model parameters
        self.set_parameters(parameters=parameters, config=config)

        # It's going to be difficult to get the loss values on the validation set
        return 0.0, 0, {}

    def setup_client(self, config: Config) -> None:
        print("Setting Up Client")
        # Save a plans file on the client to use for training
        self.gen_local_plans(config=config)  # This function also ensures plans and data identifiers are set

        # Ensure dataset fingerprint has been extracted
        fingerprint_path = join(nnUNet_preprocessed, self.dataset_name, "dataset_fingerprint.json")
        if self.always_preprocess or not exists(fingerprint_path):
            extract_fingerprints(dataset_ids=[self.dataset_id])

        # Ensure a copy of the dataset json is in the preprocessing folder
        if self.always_preprocess or not exists(join(nnUNet_preprocessed, self.dataset_name, "dataset.json")):
            shutil.copy(
                join(nnUNet_raw, self.dataset_name, "dataset.json"),
                join(nnUNet_preprocessed, self.dataset_name, "dataset.json"),
            )

        # Ensure preprocessed data is ready
        pp_data_path = join(
            nnUNet_preprocessed, self.dataset_name, str(self.data_identifier) + "_" + str(config["nnunet_config"])
        )
        if self.always_preprocess or not exists(pp_data_path):
            preprocess_dataset(
                dataset_id=self.dataset_id,
                plans_identifier=self.plans_identifier,
                configurations=[config["nnunet_config"]],
                num_processes=self.get_num_processes([config["nnunet_config"]]),
            )

        # Set the ckpt path where the model params will be stored
        self.ckpt = join(
            nnUNet_results,
            self.dataset_name,
            str(config["nnunet_trainer"]) + "__" + str(self.plans_identifier) + "__" + str(config["nnunet_config"]),
            "fold_" + str(config["fold"]),
            "checkpoint_final.pth",
        )

        self.initialized = True

    def gen_local_plans(self, config: Config) -> None:
        print("Setting Up Client")
        # Get the nnunet plans specified by the server
        plans = pickle.loads(config["nnunet_plans"])  # type: ignore

        # Get the dataset json of the local client dataset
        dataset_json = load_json(join(nnUNet_raw, self.dataset_name, "dataset.json"))

        # Change plans name
        if self.plans_identifier is None:
            self.plans_identifier = f"FL-Dataset{self.dataset_id:03d}" + "-" + plans["plans_name"]
        plans["plans_name"] = self.plans_identifier

        # Change dataset name
        plans["dataset_name"] = self.dataset_name

        # Change data identifier and ensure batch size is within limits
        if self.data_identifier is None:
            self.data_identifier = self.plans_identifier
        num_samples = dataset_json["numTraining"]
        bs_5percent = round(num_samples * 0.05)  # Set max batch size to 5 percent of dataset
        for c in plans["configurations"].keys():
            if "data_identifier" in plans["configurations"][c].keys():
                plans["configurations"][c]["data_identifier"] = self.data_identifier + "_" + c

            if "batch_size" in plans["configurations"][c].keys():
                old_bs = plans["configurations"][c]["batch_size"]
                new_bs = max(min(old_bs, bs_5percent), 2)  # Min 2, max 5 percent of dataset
                plans["configurations"][c]["batch_size"] = new_bs

        # Save the modified plans on the client
        if not exists(join(nnUNet_preprocessed, self.dataset_name)):
            os.makedirs(join(nnUNet_preprocessed, self.dataset_name))
        plans_save_path = join(nnUNet_preprocessed, self.dataset_name, self.plans_identifier + ".json")
        save_json(plans, plans_save_path, sort_keys=False)

    def check_config(self, config: Config) -> None:
        required_keys = ["local_epochs", "nnunet_plans", "nnunet_config", "fold", "nnunet_trainer"]

        for key in required_keys:
            if key not in config.keys():
                raise ValueError(f"Missing key {key} in server config")

    def get_num_processes(self, nnunet_configs: list) -> list:
        default_num_processes = {"2d": 4, "3d_lowres": 8, "3d_fullres": 4}
        return [default_num_processes[c] if c in default_num_processes.keys() else 4 for c in nnunet_configs]

    def get_num_val(self, config: Config) -> int:
        splits = load_json(join(nnUNet_preprocessed, self.dataset_name, "splits_final.json"))
        return len(splits[config["fold"]]["val"])

    def get_num_training(self, config: Config) -> int:
        splits = load_json(join(nnUNet_preprocessed, self.dataset_name, "splits_final.json"))
        return len(splits[config["fold"]]["train"])

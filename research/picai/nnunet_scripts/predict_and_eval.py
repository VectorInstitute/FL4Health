import argparse
import json
from os.path import join
from typing import List

import nnunetv2
import numpy as np
import torch
import yaml
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager


def get_predictor(ckpt_list: List[str], nnunet_config: str, dataset_json: dict, plans: dict) -> nnUNetPredictor:
    """
    Returns an initialized nnUNetPredictor for a set of nnunet models with the
    same config and architecture

    Args:
        ckpt_list (List[str]): A list containing the paths to the checkpoint
            files for the nnunet models
        nnunet_config (str): The nnunet config of the the models specifiec in
            ckpt_list.
        dataset_json (dict): The dataset json dict that specifies the label
            structure for all the models. The dataset json for the training set
            will suffice
        plans (dict): The nnunet plans used for the models during training.
            Contains important information about data preprocessing.

    Returns:
        nnUNetPredictor: _description_
    """
    # Get parameters for each model
    parameters = []
    trainer_name = None
    inference_allowed_mirror_axes = False
    for ckpt in ckpt_list:
        model = torch.load(ckpt)
        if isinstance(model, dict):
            parameters.append(model["network_weights"])
            # Check to see if we can get additional info from checkpoint
            if "trainer_name" in model.keys():
                trainer_name = model["trainer_name"]
            if "inference_allowed_mirroring_axes" in model.keys():
                inference_allowed_mirror_axes = model["inference_allowed_mirroring_axes"]
        elif isinstance(model, torch.nn.Module):
            parameters.append(model.state_dict())

    if trainer_name is None:
        # Use default trainer class
        trainer_name = "nnUNetTrainer"

    plans_manager = PlansManager(plans)
    config_manager = plans_manager.get_configuration(nnunet_config)
    num_input_channels = determine_num_input_channels(plans_manager, config_manager, dataset_json)
    trainer_class = recursive_find_python_class(
        join(nnunetv2.__path__[0], "training", "nnUNetTrainer"), trainer_name, "nnunetv2.training.nnUNetTrainer"
    )
    network = trainer_class.build_network_architecture(
        config_manager.network_arch_class_name,
        config_manager.network_arch_init_kwargs,
        config_manager.network_arch_init_kwargs_req_import,
        num_input_channels,
        plans_manager.get_label_manager(dataset_json).num_segmentation_heads,
        enable_deep_supervision=False,  # We want this turned off for inference
    )

    predictor = nnUNetPredictor()
    predictor.manual_initialization(
        network=network,
        plans_manager=plans_manager,
        configuration_manager=config_manager,
        parameters=parameters,
        dataset_json=dataset_json,
        trainer_name=trainer_name,
        inference_allowed_mirroring_axes=inference_allowed_mirror_axes,
    )
    return predictor


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="nnunet predictor",
        description="""Runs inference on raw input data given a number of
            compatible nnunet models. Then evaluated the predictions using
            picai_eval""",
        epilog="""The predictions from models of the same nnunet config are
            averaged first, then the averaged predictions from each different
            nnunet config are averaged to provide a final prediction. This
            script can be used with only a single config and even a single
            model""",
    )
    parser.add_argument(
        "--config-file",
        required=True,
        type=str,
        help="""Path to a yaml config file. The three required keys are plans,
            dataset_json and one or more nnunet_configs (eg. 2d, 3d_fullres
            etc.). The nnunet config keys should contain a list of string
            filepaths to one or more model checkpoint files. The model
            checkpoints can be dicts with the 'network_weights' key or
            nn.Modules. The plans key should be the path to the nnunet model
            plans json file. The dataset_json key should be the path to the
            dataset json of the training dataset.""",
    )
    parser.add_argument(
        "--input-data",
        required=True,
        type=str,
        help="""Path to the folder containing the raw input data that has not
        been processed by nnunet yet""",
    )
    parser.add_argument(
        "--output-folder",
        required=False,
        type=str,
        help="""[OPTIONAL] Path to the output folder to save the model
            predicted probabilities. If not provided the probabilities are not
            saved""",
    )

    args = parser.parse_args()

    config = yaml.safe_load(open(args.config_file, "r"))
    dataset_json = json.load(open(config["dataset_json"], "r"))
    plans = json.load(open(config["plans"], "r"))

    # Need to ensure this works for a single model and single config
    config_probs_list = []
    for key in config.keys():
        if key in ["2d", "3d_fullres", "3d_lowres", "3d_cascade_fullres"]:
            # Get predictor for config
            predictor = get_predictor(
                ckpt_list=config[key], nnunet_config=str(key), dataset_json=dataset_json, plans=plans
            )
            # Get predicted annotations and probabilities
            # Setting output folder to None changes the behaviour of function
            # to return the outputs instead of saving them
            preds = predictor.predict_from_files(
                list_of_lists_or_source_folder=args.input_data,
                output_folder_or_list_of_truncated_output_files=None,
                save_probabilities=True,
            )
            # returns a list with shape [num_samples, 2] where the second
            # dimension contains the predicted annotation as the first element
            # and the predicted probabilities as the second element. Outputs
            # are numpy arrays

            # Convert list to tensors
            probs = []
            for pred in preds:
                probs.append(pred[1])

            # Each element of config_probs_list will have
            # shape (num_samples, num_classes, spatial_dims...)
            config_probs_list.append(np.stack(probs))
            del probs
            del preds

    # If only one element stack adds an empty dim
    final_preds = np.mean(np.stack(config_probs_list), axis=0)
    # final preds shape: (num_samples, num_classes, spatial_dims...)
    print(np.shape(final_preds))


if __name__ == "__main__":
    main()

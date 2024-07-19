import argparse
import contextlib
import json
import os
import time
import warnings
from logging import INFO
from os.path import isdir, join
from typing import List, Tuple

import nnunetv2
import numpy as np
import torch
import yaml
from flwr.common.logger import log

with warnings.catch_warnings():
    # We get a bunch of scipy deprecation warnings from these packages
    # Curiosly this only happens if flwr is imported first
    # Raised issue https://github.com/MIC-DKFZ/nnUNet/issues/2370
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
    from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
    from nnunetv2.utilities.utils import create_lists_from_splitted_dataset_folder
    from numpy.typing import NDArray


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

    # Helper function to make code cleaner
    def check_for_ckpt_info(model: dict) -> Tuple[str, bool]:
        """
        Checks model dict for trainer name and inference_allowed_mirroring_axes

        Returns:
            Tuple[Optional[str], bool]: Tuple with elements trainer_name and
                inference_allowed_mirroring_axes. Defaults to
                ('nnUNetTrainer, False)
        """
        trainer_name = "nnUNetTrainer"
        inference_allowed_mirror_axes = False
        if "trainer_name" in model.keys():
            trainer_name = model["trainer_name"]
        if "inference_allowed_mirroring_axes" in model.keys():
            inference_allowed_mirror_axes = model["inference_allowed_mirroring_axes"]

        return trainer_name, inference_allowed_mirror_axes

    # Create unintialized predictor instance
    predictor = nnUNetPredictor(verbose=False, verbose_preprocessing=False, allow_tqdm=False)

    # Get parameters for each model and maybe some predictor init parameters
    trainer_name: str = "nnUNetTrainer"  # Default trainer class
    mirror_axes = False
    parameters = []
    for ckpt in ckpt_list:
        if isdir(ckpt):  # Assume this is an nnunet results directory
            with contextlib.redirect_stdout(open(os.devnull, "w")):
                folds = predictor.auto_detect_available_folds(ckpt, "checkpoint_best.pth")
            for fold in folds:
                f = int(fold) if fold != "all" else fold
                model = torch.load(join(ckpt, f"fold_{f}", "checkpoint_best.pth"), map_location=torch.device("cpu"))
                # nnunet saves their models as dicts always
                parameters.append(model["network_weights"])
                trainer_name, mirror_axes = check_for_ckpt_info(model)

        else:
            model = torch.load(ckpt, map_location=torch.device("cpu"))
            if isinstance(model, dict):
                parameters.append(model["network_weights"])
                # Check to see if we can get additional info from checkpoint
                trainer_name, mirror_axes = check_for_ckpt_info(model)

            elif isinstance(model, torch.nn.Module):
                parameters.append(model.state_dict())

    # Get the rest of the objects needed for predictor initialization
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

    predictor.manual_initialization(
        network=network,
        plans_manager=plans_manager,
        configuration_manager=config_manager,
        parameters=parameters,
        dataset_json=dataset_json,
        trainer_name=trainer_name,
        inference_allowed_mirroring_axes=mirror_axes,
    )
    return predictor


def predict(config_path: str, input_data: str, output_folder: str) -> NDArray:
    # Load config and nnunet required dicts
    config = yaml.safe_load(open(config_path, "r"))
    dataset_json = json.load(open(config["dataset_json"], "r"))
    plans = json.load(open(config["plans"], "r"))

    # Convert input folder into a list of filenames so that we know which
    # output preds correspond to which input files
    input_data = create_lists_from_splitted_dataset_folder(folder=input_data, file_ending=dataset_json["file_ending"])
    case_identifiers = [case[0].split(".")[0][:-5] for case in input_data]

    model_count = 0
    config_probs_list = []
    for key in config.keys():
        if key in ["2d", "3d_fullres", "3d_lowres", "3d_cascade_fullres"]:
            # Get predictor for config
            predictor = get_predictor(
                ckpt_list=config[key], nnunet_config=str(key), dataset_json=dataset_json, plans=plans
            )
            n_models = len(predictor.list_of_parameters)
            log(INFO, f"Found {n_models} models for nnunet config {key}. Starting Inference")
            model_count += n_models
            t = time.time()

            # Silence stdout because predictor still prints stuff
            with contextlib.redirect_stdout(open(os.devnull, "w")):
                # Get predicted annotations and probabilities
                # Setting output folder to None changes the behaviour of
                # function to return the outputs instead of saving them
                preds = predictor.predict_from_files(
                    list_of_lists_or_source_folder=input_data,
                    output_folder_or_list_of_truncated_output_files=None,
                    save_probabilities=True,
                )
            secs = time.time() - t
            log(INFO, f"Inference complete: {secs:.1f}s total, {secs/(len(case_identifiers)*n_models):.1f}s/case")
            log(INFO, "")
            # preds is shape [num_samples, 2] where the second dimension's
            # first element is the predicted annotation and the second element
            # is the predicted probabilities. Outputs are numpy arrays

            # Get just the probabilities
            probs = []
            for pred in preds:
                probs.append(pred[1])

            # Each element of config_probs_list will have
            # shape (num_samples, num_classes, spatial_dims...)
            # If only one element stack adds an empty dim
            config_probs_list.append(np.stack(probs))
            del probs
            del preds

    # If only one element stack adds an empty dim
    final_preds = np.mean(np.stack(config_probs_list), axis=0)

    # Logs
    shape = np.shape(final_preds)
    log(
        INFO,
        (
            f"Finished running inference with {model_count} models on "
            f"{shape[0]} samples. Number of output classes: {shape[1]}"
        ),
    )
    log(
        INFO,
        f"Final Predictions Array Shape: {shape}",
    )

    # final preds shape: (num_samples, num_classes, spatial_dims...)
    return final_preds


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="nnunet predictor",
        description="""Runs inference on raw input data given a number of
            compatible nnunet models. Then evaluates the predictions using
            picai_eval""",
        epilog="""The predictions from models of the same nnunet config are
            averaged first, then the averaged predictions from each different
            nnunet config are averaged to provide a final prediction.
            Regardless of the number of models and or nnunet configs, this
            script always produces only a single final prediction for each
            input image. This script can be used with only a single config and
            even a single model""",
    )
    parser.add_argument(
        "--config-path",
        required=True,
        type=str,
        help="""Path to a yaml config file. The three required keys are plans,
            dataset_json and one or more nnunet_configs (eg. 2d, 3d_fullres
            etc.). The nnunet config keys should contain a list of paths. If
            the path points to a file it should be a model checkpoint. The model
            checkpoints can be dicts with the 'network_weights' key or
            nn.Modules. If the path points to a directory it should be an
            nnunet results folder for a particular dataset-config-trainer
            combo. The plans key should be the path to the nnunet model
            plans json file. The dataset_json key should be the path to the
            dataset json of one of the training datasets. Or create a new json
            yourself with the 'label' and 'file_ending' keys and their
            corresponding values as specified by nnunet""",
    )
    parser.add_argument(
        "--input-data",
        required=True,
        type=str,
        help="""Path to the folder containing the raw input data that has not
        been processed by nnunet yet. File names must follow the nnunet
        convention where each channel modality is stored as a seperate file.
        File names should be case-identifier_0000 where 0000 is a 4 digit
        integer representing the channel/modality of the image. All cases must
        have the same number of channels N numbered from 0 to N""",
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

    predict(args.config_path, args.input_data, args.output_folder)


if __name__ == "__main__":
    main()

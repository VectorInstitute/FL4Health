import argparse
import contextlib
import json
import os
import shutil
import time
import warnings
from logging import INFO
from os.path import basename, exists, isdir, join

import numpy as np
import torch
import yaml
from flwr.common.logger import log

from research.picai.fl_nnunet.nnunet_utils import NnunetConfig


with warnings.catch_warnings():
    # We get a bunch of scipy deprecation warnings from these packages
    # Curiously this only happens if flwr is imported first
    # Raised issue https://github.com/MIC-DKFZ/nnUNet/issues/2370
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    import nnunetv2
    from nnunetv2.ensembling.ensemble import ensemble_folders
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
    from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
    from nnunetv2.utilities.utils import create_lists_from_splitted_dataset_folder


def yaml_join(loader: yaml.Loader, node: yaml.SequenceNode) -> str:
    seq = loader.construct_sequence(node)
    return os.path.join(*seq)


def get_predictor(ckpt_list: list[str], nnunet_config: str, dataset_json: dict, plans: dict) -> nnUNetPredictor:
    """
    Returns an initialized nnUNetPredictor for a set of nnunet models with the same config and architecture.

    Args:
        ckpt_list (list[str]): A list containing the paths to the checkpoint files for the nnunet models
        nnunet_config (str): The nnunet config of the the models specific in ckpt_list.
        dataset_json (dict): The dataset json dict that specifies the label structure for all the models. The dataset
            json for the training set will suffice
        plans (dict): The nnunet plans used for the models during training. Contains important information about data
            preprocessing.

    Returns:
        (nUNetPredictor): An ``nnUNetPredictor`` class for the set of models specified by the ckpt_list.
    """

    # Helper function to make code cleaner
    def check_for_ckpt_info(model: dict) -> tuple[str, bool]:
        """
        Checks model dict for trainer name and ``inference_allowed_mirroring_axes``.

        Returns:
            (tuple[str | None, bool]): Tuple with elements ``trainer_name`` and `inference_allowed_mirroring_axes`.
                Defaults to ("nnUNetTrainer", False).
        """
        trainer_name = "nnUNetTrainer"
        inference_allowed_mirror_axes = False
        if "trainer_name" in model:
            trainer_name = model["trainer_name"]
        if "inference_allowed_mirroring_axes" in model:
            inference_allowed_mirror_axes = model["inference_allowed_mirroring_axes"]

        return trainer_name, inference_allowed_mirror_axes

    # Create uninitialized predictor instance
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
                model = torch.load(
                    join(ckpt, f"fold_{f}", "checkpoint_best.pth"),
                    map_location=torch.device("cpu"),
                    weights_only=False,
                )
                # nnunet saves their models as dicts always
                parameters.append(model["network_weights"])
                trainer_name, mirror_axes = check_for_ckpt_info(model)

        else:
            model = torch.load(ckpt, map_location=torch.device("cpu"), weights_only=False)
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


def predict(
    config_path: str,
    input_folder: str,
    output_folder: str,
    probs_folder_name: str = "predicted_probability_maps",
    annotations_folder_name: str = "predicted_annotations",
    verbose: bool = True,
) -> None:
    """
    Uses multiprocessing to quickly do model inference for a single model, a group of models with the same nnunet
    config or an ensemble of different nnunet configs each with one or more models.

    Args:
        config_path (str): Path to a yaml config file. The three required keys are plans, dataset_json and one or
            more nnunet_configs (e.g. 2d, 3d_fullres etc.). The nnunet config keys should contain a list of paths. If
            the path points to a file it should be a model checkpoint. The model checkpoints can be dicts with the
            'network_weights' key or nn.Modules. If the path points to a directory it should be an nnunet results
            folder for a particular dataset-config-trainer combo. The plans key should be the path to the nnunet
            model plans json file. The dataset_json key should be the path to the dataset json of one of the training
            datasets. Or create a new json yourself with the 'label' and 'file_ending' keys and their corresponding
            values as specified by nnunet. A !join constructor that maps to os.path.join has been defined when loading
            the config to allow the user to make their configs more readable. Eg.
                base_path: &base_path /home/user/data
                dataset_json: !join [*base_path, 'PICAI', 'dataset.json']
        input_folder (str): Path to the folder containing the raw input data that has not been processed by nnunet
            yet. File names must follow the nnunet convention where each channel modality is stored as a separate file.
            File names should be case-identifier_0000 where 0000 is a 4 digit integer representing the
            channel/modality of the image. All cases must have the same number of channels N numbered from 0 to N.
        output_folder (str): Path to save the predicted probabilities and predicted annotations. Each will be stored
            in a separate subdirectory. Probabilities will be stored as .npz files. The NPZ file object will have the
            key 'probabilities'. The predicted annotations will be saved as the original input image file format
        probs_folder_name (str, optional): What to name the folder within the output folder that the probabilities
            will be stored in. Defaults to "predicted_probability_maps".
        annotations_folder_name (str, optional): What to name the folder within the output folder that the predicted
            annotations will be stored in. Defaults to "predicted_annotations".
        verbose (bool, optional): Setting this to false will limit the amount of logging produced by this function.
            Defaults to True.
    """
    # Note: I should split output folder into two separate paths for model outputs
    t_start = time.time()

    # Add !join constructor to yaml so that config files can be more readable
    yml_loader = yaml.SafeLoader
    yml_loader.add_constructor("!join", yaml_join)

    # Load config and nnunet required dicts
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yml_loader)
    with open(config["dataset_json"], "r") as f:
        dataset_json = json.load(f)
    with open(config["plans"], "r") as f:
        plans = json.load(f)

    # Get case identifiers
    input_files = create_lists_from_splitted_dataset_folder(
        folder=input_folder, file_ending=dataset_json["file_ending"]
    )
    case_identifiers = [basename(case[0]).split(".")[0][:-5] for case in input_files]
    num_samples = len(case_identifiers)

    # Model inference
    model_count = 0
    cfg_folders = []
    for key in config:
        if key in [cfg.value for cfg in NnunetConfig]:
            # Get predictor for config
            predictor = get_predictor(
                ckpt_list=config[key], nnunet_config=str(key), dataset_json=dataset_json, plans=plans
            )
            n_models = len(predictor.list_of_parameters)
            if verbose:
                log(INFO, f"Found {n_models} models for nnunet config {key}. Starting Inference")
            model_count += n_models
            t = time.time()

            # Create temporary output folder and add it to list
            cfg_output_folder = join(output_folder, key)
            cfg_folders.append(cfg_output_folder)

            # Silence stdout because predictor still prints stuff
            with contextlib.redirect_stdout(open(os.devnull, "w")):
                predictor.predict_from_files(
                    list_of_lists_or_source_folder=input_folder,
                    output_folder_or_list_of_truncated_output_files=cfg_output_folder,
                    save_probabilities=True,
                )

            # Logging
            secs = time.time() - t
            if verbose:
                log(INFO, f"Inference complete: {secs:.1f}s total, {secs / (num_samples * n_models):.1f}s/case")
                log(INFO, "")

    # Now we need to ensemble the predictions from each config
    if verbose:
        log(INFO, "Ensembling predictions...")
    t = time.time()
    ensemble_folders(
        list_of_input_folders=cfg_folders,
        output_folder=join(output_folder, annotations_folder_name),
        save_merged_probabilities=True,
        dataset_json_file_or_dict=dataset_json,
        plans_json_file_or_dict=plans,
    )
    secs = time.time() - t
    if verbose:
        log(INFO, f"Ensembling complete in {secs:.1f}s, {secs / num_samples:.1f}s/case")
        log(INFO, "")

    if verbose:
        log(INFO, "Rearranging files...")
    t = time.time()

    # Copy some metadata files into the output directory
    config_name = basename(config_path)
    with open(join(output_folder, config_name), "w") as f:
        yaml.dump(config, f, sort_keys=False, indent=4)

    shutil.copy(  # Data properties should be the same for all input images
        src=join(cfg_folders[0], case_identifiers[0] + ".pkl"), dst=join(output_folder, "data_properties.pkl")
    )

    plans_name = basename(config["plans"])
    with open(join(output_folder, plans_name), "w") as f:
        json.dump(plans, f, indent=4, sort_keys=False)

    os.replace(join(output_folder, annotations_folder_name, "dataset.json"), join(output_folder, "dataset.json"))

    # Remove the individual predictions for each model
    for folder in cfg_folders:
        shutil.rmtree(folder)

    # Move predicted probabilities
    if not exists(join(output_folder, probs_folder_name)):
        os.makedirs(join(output_folder, probs_folder_name))
    for case in case_identifiers:
        # Ensemble method saves two copies of probabilities for some reason
        # We'll keep the numpy compressed one and delete the serialized one
        os.replace(
            join(output_folder, annotations_folder_name, f"{case}.npz"),
            join(output_folder, probs_folder_name, f"{case}.npz"),
        )
        os.remove(join(output_folder, annotations_folder_name, f"{case}.pkl"))

    secs = time.time() - t
    if verbose:
        log(INFO, f"File management complete in {secs:.1f}s")
        log(INFO, "")

    # Logs
    sample = np.load(join(output_folder, probs_folder_name, f"{case_identifiers[0]}.npz"))
    shape = sample["probabilities"].shape
    if verbose:
        log(
            INFO,
            (f"Finished running inference with {model_count} models on {num_samples} cases."),
        )
        log(INFO, f"\tNum Cases: {num_samples}")
        log(INFO, f"\tNum Classes: {shape[0]}")
        log(INFO, f"\tSpatial Dimensions {shape[1:]}")
        secs = time.time() - t_start
        log(INFO, f"Total Time: {secs:.1f}s ({secs / num_samples:.1f}s/case)")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="nnunet predictor",
        description="""Runs inference on raw input data given a number of
            compatible nnunet models.""",
        epilog="""The predictions from models of the same nnunet config are
            averaged first, then the averaged predictions from each different
            nnunet config are averaged to provide a final prediction.
            Regardless of the number of models and or nnunet configs, this
            script always produces only a single final prediction for each
            input image. This script can be used with only a single config and
            even a single model.""",
    )
    parser.add_argument(
        "--config-path",
        required=True,
        type=str,
        help="""Path to a yaml config file. The three required keys
            are plans, dataset_json and one or more nnunet_configs (e.g. 2d,
            3d_fullres etc.). The nnunet config keys should contain a list of
            paths. If the path points to a file it should be a model
            checkpoint. The model checkpoints can be dicts with the
            'network_weights' key or nn.Modules. If the path points to a
            directory it should be an nnunet results folder for a particular
            dataset-config-trainer combo. The plans key should be the path to
            the nnunet model plans json file. The dataset_json key should be
            the path to the dataset json of one of the training datasets. Or
            create a new json yourself with the 'label' and 'file_ending' keys
            and their corresponding values as specified by nnunet. A !join
            constructor that maps to os.path.join has been defined when
            loading the config to allow the user to make their configs more
            readable. Eg.
                    base_path: &base_path /home/user/data
                    dataset_json: !join [*base_path, 'PICAI', 'dataset.json']""",
    )
    parser.add_argument(
        "--input-folder",
        required=True,
        type=str,
        help="""Path to the folder containing the raw input data that has not
            been processed by nnunet yet. File names must follow the nnunet
            convention where each channel modality is stored as a separate
            file. File names should be case-identifier_0000 where 0000 is a 4
            digit integer representing the channel/modality of the image. All
            cases must have the same N channels numbered from 0 to N.""",
    )
    parser.add_argument(
        "--output-folder",
        required=True,
        type=str,
        help="""[OPTIONAL] Path to save the predicted probabilities and
            predicted annotations. Each will be stored in a separate
            subdirectory. Probabilities will be stored as .npz files.
            The NPZ file object will have the key 'probabilities'. The
            predicted annotations will be saved as the original input image
            file format""",
    )

    args = parser.parse_args()

    predict(args.config_path, args.input_folder, args.output_folder)


if __name__ == "__main__":
    main()

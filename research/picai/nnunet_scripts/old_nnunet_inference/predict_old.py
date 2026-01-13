import argparse
import contextlib
import json
import multiprocessing
import os
import time
import warnings
from collections.abc import Generator
from logging import INFO
from os.path import basename, isdir, join
from typing import Any

import numpy as np
import torch
import yaml
from flwr.common.logger import log
from numpy.typing import NDArray


with warnings.catch_warnings():
    # We get a bunch of scipy deprecation warnings from these packages
    # Curiously this only happens if flwr is imported first
    # Raised issue https://github.com/MIC-DKFZ/nnUNet/issues/2370
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    import nnunetv2
    from nnunetv2.configuration import default_num_processes
    from nnunetv2.inference.export_prediction import convert_predicted_logits_to_segmentation_with_correct_shape
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    from nnunetv2.utilities.file_path_utilities import check_workers_alive_and_busy
    from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
    from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
    from nnunetv2.utilities.utils import create_lists_from_splitted_dataset_folder


class MyNnUNetPredictor(nnUNetPredictor):
    def predict_from_data_iterator(
        self, data_iterator: Generator, return_probabilities: bool = False, num_processes: int = default_num_processes
    ) -> dict[str, Any]:
        """
        Override of the predict from data iterator class so that we can have it return the model outputs along with
        their output filenames and data properties. The parent class method either saves the data and returns
        nothing, or does not save the data and only returns the model outputs. We are going to change as little as
        possible. This function is based off of nnunetv2 version 2.4.2.

        Args:
            data_iterator (Generator): The data iterator
            return_probabilities (bool, optional): Whether or not to return the predicted probability maps.
                Defaults to False.
            num_processes (int, optional): The number of processes to use when exporting predictions.
                Defaults to nnunet's default number of processes (at time of writing this is 8)

        Returns:
            (List): A dictionary containing the predicted annotations and the data properties for each file. The
                dictionary may also contain the predicted probabilities and the output file names associated with each
                sample.
        """
        with multiprocessing.get_context("spawn").Pool(num_processes) as pool:
            # Have to ignore errors when defining worker list because mypy
            # doesn't understand multiprocessing.get_context
            worker_list = list(pool._pool)  # type: ignore
            model_outputs: Any = []
            ofiles = []
            properties_list = []

            for preprocessed in data_iterator:
                data = preprocessed["data"]
                if isinstance(data, str):
                    delfile = data
                    data = torch.from_numpy(np.load(data))
                    os.remove(delfile)

                if preprocessed["ofile"] is not None:
                    ofiles.append(preprocessed["ofile"])

                properties = preprocessed["data_properties"]
                properties_list.append(properties)

                # let's not get into a runaway situation where the GPU predicts
                # so fast that the disk has to b swamped with npy files
                proceed = not check_workers_alive_and_busy(pool, worker_list, model_outputs, allowed_num_queued=2)
                while not proceed:
                    time.sleep(0.1)
                    proceed = not check_workers_alive_and_busy(pool, worker_list, model_outputs, allowed_num_queued=2)

                logits = self.predict_logits_from_preprocessed_data(data).cpu()

                # Send prediction off to background worker for resampling
                model_outputs.append(
                    pool.starmap_async(
                        convert_predicted_logits_to_segmentation_with_correct_shape,
                        (
                            (
                                logits,
                                self.plans_manager,
                                self.configuration_manager,
                                self.label_manager,
                                properties,
                                return_probabilities,
                            ),
                        ),
                    ),
                )

            # Package outputs as dictionary
            return_dict = {"annotation_preds": [], "data_properties": properties_list}

            if len(ofiles) == len(model_outputs):
                return_dict["ofiles"] = ofiles

            if return_probabilities:
                return_dict["probability_preds"] = []

            for output in model_outputs:
                data = output.get()[0]
                if return_probabilities:
                    return_dict["annotation_preds"].append(data[0])
                    return_dict["probability_preds"].append(data[1])
                else:
                    return_dict["annotation_preds"].append(data)

        return return_dict


def get_predictor(ckpt_list: list[str], nnunet_config: str, dataset_json: dict, plans: dict) -> nnUNetPredictor:
    """
    Returns an initialized nnUNetPredictor for a set of nnunet models with the
    same config and architecture.

    Args:
        ckpt_list (list[str]): A list containing the paths to the checkpoint
            files for the nnunet models
        nnunet_config (str): The nnunet config of the the models specific in
            ckpt_list.
        dataset_json (dict): The dataset json dict that specifies the label
            structure for all the models. The dataset json for the training set
            will suffice
        plans (dict): The nnunet plans used for the models during training.
            Contains important information about data preprocessing.

    Returns:
        (MyNnUNetPredictor): A subclass of the nnUNetPredictor class for the set
            of models specified by the ckpt_list. The subclasses only
            difference is that it returns a dictionary with more information
            as opposed to just a list of numpy arrays.
    """

    # Helper function to make code cleaner
    def check_for_ckpt_info(model: dict) -> tuple[str, bool]:
        """
        Checks model dict for trainer name and ``inference_allowed_mirroring_axes``.

        Returns:
            (tuple[str | None, bool]): Tuple with elements trainer_name and ``inference_allowed_mirroring_axes``.
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
    predictor = MyNnUNetPredictor(verbose=False, verbose_preprocessing=False, allow_tqdm=False)

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
    probs_folder: str | None = None,
    annotations_folder: str | None = None,
    verbose: bool = True,
) -> tuple[NDArray, NDArray, list[str]]:
    """
    Uses multiprocessing to quickly do model inference for a single model, a group of models with the same nnunet
    config or an ensemble of different nnunet configs each with one or more models.

    Args:
        config_path (str): Path to a yaml config file. The three required keys are plans, dataset_json and one or more
            nnunet_configs (e.g. 2d, 3d_fullres etc.). The nnunet config keys should contain a list of paths. If the
            path points to a file it should be a model checkpoint. The model checkpoints can be dicts with the
            'network_weights' key or nn.Modules. If the path points to a directory it should be an nnunet results
            folder for a particular dataset-config-trainer combo. The plans key should be the path to the nnunet model
            plans json file. The dataset_json key should be the path to the dataset json of one of the training
            datasets. Or create a new json yourself with the 'label' and 'file_ending' keys and their
            corresponding values as specified by nnunet
        input_folder (str): Path to the folder containing the raw input data that has not been processed by nnunet
            yet. File names must follow the nnunet convention where each channel modality is stored as a separate
            file. File names should be case-identifier_0000 where 0000 is a 4 digit integer representing the
            channel/modality of the image. All cases must have the same number of channels N numbered from 0 to N.
        probs_folder (str | None, optional): Path to the output folder to save the model predicted probabilities. If
            not provided the probabilities are not saved. Defaults to None.
        annotations_folder (str | None, optional):Path to the output folder to save the model predicted annotations.
            Defaults to None.
        verbose (bool, optional): Setting this to false will limit the amount of logging produced by this function.
            Defaults to True.

    Returns:
        (tuple[NDArray, NDArray, list[str]]): A numpy array with a single predicted probability map for each input
            image. Shape:  (num_samples, num_classes, ...). A numpy array with a single predicted annotation map for
            each input image. Unlike the predicted  probabilities these are NOT one hot encoded.
            Shape: (num_samples, spatial_dims...). A list containing the unique case identifier for each prediction.
    """
    t_start = time.time()
    # Load config and nnunet required dicts
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    with open(config["dataset_json"], "r") as f:
        dataset_json = json.load(f)
    with open(config["plans"], "r") as f:
        plans = json.load(f)

    # Convert input folder into a list of filenames so that we know which
    # output preds correspond to which input files
    input_data = create_lists_from_splitted_dataset_folder(
        folder=input_folder, file_ending=dataset_json["file_ending"]
    )
    case_identifiers = [basename(case[0]).split(".")[0][:-5] for case in input_data]

    # Get output filelist
    if probs_folder:
        output_filelist = [join(probs_folder, case) for case in case_identifiers]
    elif annotations_folder:
        output_filelist = [join(annotations_folder, case) for case in case_identifiers]
    else:
        output_filelist = None

    # Model inference
    model_count = 0
    config_probs_list = []
    for key in config:
        if key in ["2d", "3d_fullres", "3d_lowres", "3d_cascade_fullres"]:
            # Get predictor for config
            predictor = get_predictor(
                ckpt_list=config[key], nnunet_config=str(key), dataset_json=dataset_json, plans=plans
            )
            n_models = len(predictor.list_of_parameters)
            if verbose:
                log(INFO, f"Found {n_models} models for nnunet config {key}. Starting Inference")
            model_count += n_models
            t = time.time()

            # Silence stdout because predictor still prints stuff
            with contextlib.redirect_stdout(open(os.devnull, "w")):
                # Get predicted annotations and probabilities
                # Setting output folder to None changes the behaviour of
                # function to return the outputs instead of saving them
                # UPDATE: I overrode a method so that outputs are always returned
                output = predictor.predict_from_files(
                    list_of_lists_or_source_folder=input_data,
                    output_folder_or_list_of_truncated_output_files=output_filelist,
                    save_probabilities=True,
                )

            secs = time.time() - t
            if verbose:
                log(
                    INFO,
                    f"Inference complete: {secs:.1f}s total, {secs / (len(case_identifiers) * n_models):.1f}s/case",
                )
                log(INFO, "")

            # Each element of config_probs_list will have
            # shape (num_samples, num_classes, spatial_dims...)
            # If only one element stack adds an empty dim
            config_probs_list.append(np.stack(output["probability_preds"]))

            # Save stuff for annotations
            label_manager = predictor.label_manager
            annot_writer = predictor.plans_manager.image_reader_writer_class()
            data_properties = output["data_properties"]

            # Delete variables we don't need from memory
            del output
            del predictor

    # If only one element stack adds an empty dim
    final_probs = np.mean(np.stack(config_probs_list), axis=0, dtype=float)

    annotations = []
    for prob in final_probs:
        annotations.append(label_manager.convert_probabilities_to_segmentation(prob))
    final_annotations = np.stack(annotations).astype(int)

    # Logs
    shape = np.shape(final_probs)
    if verbose:
        log(INFO, f"Finished running inference with {model_count} models on {shape[0]} samples.")
        log(INFO, f"\tNum Samples: {shape[0]}")
        log(INFO, f"\tNum Classes: {shape[1]}")
        log(INFO, f"\tSpatial Dimensions {shape[2:]}")

    # Save predicted probabilities if output_folder was provided
    if probs_folder is not None:
        t = time.time()
        for pred, case in zip(final_probs, case_identifiers):
            ofile = join(probs_folder, case + ".npz")
            np.savez_compressed(file=ofile, probabilities=pred)
        secs = time.time() - t
        if verbose:
            log(INFO, "")
            log(
                INFO,
                (
                    f"Saved predicted probability maps to disk: {secs:.1f}s "
                    f"total, {secs / len(case_identifiers):.1f}s/case"
                ),
            )

    # Maybe save predicted annotations
    if annotations_folder is not None:
        if not os.path.exists(annotations_folder):
            os.makedirs(annotations_folder)
        for pred, case, props in zip(final_annotations, case_identifiers, data_properties):
            ofile = join(annotations_folder, case + dataset_json["file_ending"])
            annot_writer.write_seg(pred, ofile, props)
        if verbose:
            log(
                INFO,
                "Saved predicted annotations to disk",
            )

    if verbose:
        secs = time.time() - t_start
        log(INFO, "")
        log(INFO, f"Total Time: {secs:.1f}s ({secs / len(case_identifiers):.1f}s/case)")

    # final probs shape: (num_samples, num_classes, spatial_dims...)
    # final annot shape: (num_samples, spatial_dims...)
    return final_probs, final_annotations, case_identifiers


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
        help="""Path to a yaml config file. The three required keys are plans,
            dataset_json and one or more nnunet_configs (e.g. 2d, 3d_fullres
            etc.). The nnunet config keys should contain a list of paths. If
            the path points to a file it should be a model checkpoint. The
            model checkpoints can be dicts with the 'network_weights' key or
            nn.Modules. If the path points to a directory it should be an
            nnunet results folder for a particular dataset-config-trainer
            combo. The plans key should be the path to the nnunet model
            plans json file. The dataset_json key should be the path to the
            dataset json of one of the training datasets. Or create a new json
            yourself with the 'label' and 'file_ending' keys and their
            corresponding values as specified by nnunet.""",
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
        "--probs-folder",
        required=False,
        type=str,
        help="""[OPTIONAL] Path to the output folder to save the model
            predicted probabilities. If not provided the probabilities are not
            saved. Will recursively create directories as necessary if the
            probabilities folder does not exist""",
    )
    parser.add_argument(
        "--annotations-folder",
        required=False,
        type=str,
        help="""[OPTIONAL] Path to the output folder to save the model
            predicted annotations. If not provided the annotations are not
            saved. Will recursively create directories as necessary if the
            annotations folder does not exist""",
    )

    args = parser.parse_args()

    predict(args.config_path, args.input_folder, args.probs_folder, args.annotations_folder)


if __name__ == "__main__":
    main()

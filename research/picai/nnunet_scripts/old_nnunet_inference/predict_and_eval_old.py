import argparse
import os
import time
import warnings
from logging import INFO
from os.path import exists, join

import numpy as np
from flwr.common.logger import log

from research.picai.nnunet_scripts.old_nnunet_inference.eval_old import (
    get_detection_maps,
    get_picai_metrics,
    load_images_from_folder,
)
from research.picai.nnunet_scripts.old_nnunet_inference.predict_old import predict


def pred_and_eval(
    config_path: str,
    inputs_folder: str,
    labels_folder: str,
    output_folder: str | None = None,
    save_probability_maps: bool = False,
    save_detection_maps: bool = False,
    save_annotations: bool = False,
    verbose: bool = True,
) -> None:
    """
    Runs inference on raw input data given a number of compatible nnunet models. Then extracts detection maps from
    those predictions and evaluates the model using the standardPICAI evaluation metrics.

    Args:
        config_path (str): Path to a yaml config file. The three required keys are plans, dataset_json and one or more
            nnunet_configs (e.g. 2d, 3d_fullres etc.). The nnunet config keys should contain a list of paths. If the
            path points to a file it should be a model checkpoint. The model checkpoints can be dicts with the
            'network_weights' key or nn.Modules. If the path points to a directory it should be an nnunet results
            folder for a particular dataset-config-trainer combo. The plans key should be the path to the nnunet model
            plans json file. The dataset_json key should be the path to the dataset json of one of the training
            datasets. Or create a new json yourself with the 'label' and 'file_ending' keys and their corresponding
            values as specified by nnunet.
        inputs_folder (str): Path to the folder containing the raw input data that has not been processed by nnunet
            yet. File names must follow the nnunet convention where each channel modality is stored as a separate
            file. File names should be case-identifier_0000 where 0000 is a 4 digit integer representing the
            channel/modality of the image. All cases must have the same N channels numbered from 0 to N.
        labels_folder (str): Path to the folder containing the ground truth annotation maps. File names must match
            the case identifiers of the input images
        output_folder (str | None, optional): Path to the output folder. By default the only output is a
            'metrics.json' file containing the evaluation results. If left as none then nothing is saved.
        save_probability_maps (bool, optional): Whether or not to save the predicted probability maps.
            Defaults to False.
        save_detection_maps (bool, optional): Whether or not to save the predicted lesion detection maps.
            Defaults to False.
        save_annotations (bool, optional): Whether or not to save the predicted lesion annotations maps.
            Defaults to False.
        verbose (bool, optional): Whether or not to print logs to stdout. Defaults to True.
    """
    # Ensure an output folder is specified
    if save_probability_maps or save_detection_maps or save_annotations:
        assert output_folder is not None, (
            "Can not save the probability maps, annotations or the detection maps if no output folder is specified"
        )

    # Run inference and maybe save probability maps.
    # output folders will be created if they dont exist
    if save_probability_maps and output_folder is not None:
        prob_out_folder = join(output_folder, "predicted_probability_maps")
    else:
        prob_out_folder = None

    if save_annotations and output_folder is not None:
        annotations_folder = join(output_folder, "predicted_annotations")
    else:
        annotations_folder = None

    probability_preds, annotation_preds, case_identifiers = predict(
        config_path=config_path,
        input_folder=inputs_folder,
        probs_folder=prob_out_folder,
        annotations_folder=annotations_folder,
        verbose=verbose,
    )
    del annotation_preds

    # Extract lesion detection maps
    if verbose:
        log(INFO, "")
        log(INFO, "Starting lesion detection")

    t = time.time()
    detection_maps = get_detection_maps(probability_preds)
    secs = time.time() - t

    if verbose:
        log(INFO, f"Lesion detection complete: {secs:.2f}s total, {secs / len(case_identifiers):.2f}s/case")

    # Save detection maps
    if save_detection_maps and output_folder is not None:
        det_output_folder = join(output_folder, "detection_maps")
        if not exists(det_output_folder):
            os.makedirs(det_output_folder)
        t = time.time()
        for det, case in zip(detection_maps, case_identifiers):
            ofile = join(det_output_folder, case + ".npz")
            np.savez_compressed(ofile, detection_map=det)
        secs = time.time() - t
        if verbose:
            log(INFO, f"Saved detection maps to disk: {secs:.2f}s total, {secs / len(detection_maps):.1f}s/case")

    # Load the input data
    labels = load_images_from_folder(labels_folder, case_identifiers).astype(int)

    # Check if labels need to be one hot encoded
    if detection_maps.ndim != labels.ndim:
        num_classes = probability_preds.shape[1]
        labels_one_hot = (np.arange(num_classes) == labels[..., None]).astype(int)
        labels_one_hot = np.moveaxis(labels_one_hot, -1, 1)

    # Calculate Metrics
    t = time.time()
    metrics = get_picai_metrics(detection_maps, labels_one_hot)
    secs = time.time() - t

    # Print metrics
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        if verbose:
            log(INFO, "")
            log(INFO, "Evaluation Results")
            log(INFO, f"\teval took {secs:.1f}s total")
            log(INFO, f"\t{metrics}")
            log(INFO, f"\tPicai Score: {metrics.score}")

    # Save metrics
    if output_folder is not None:
        metrics.save_full(join(output_folder, "picai_eval_metrics.json"))


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="Predict and Evaluate nnUNet models or nnUNet Ensembles",
        description="""Runs inference on raw input data given a number of
            compatible nnunet models. Then extracts detection maps from
            those predictions and evaluates the model using the standard
            PICAI evaluation metrics.""",
        epilog="""The predictions from models of the same nnunet config are
            averaged first, then the averaged predictions from each different
            nnunet config are averaged to provide a final prediction.
            Regardless of the number of models and or nnunet configs, this
            script always produces only a single final prediction for each
            input image. This script can be used with only a single config and
            even a single model.

            Detection maps are extracted using the extract_lesion_candidates
            method from the report_guided_annotation api
            (https://github.com/DIAGNijmegen/Report-Guided-Annotation/tree/main)

            The PICAI evaluation metrics are computed using the picai_eval api
            (https://github.com/DIAGNijmegen/picai_eval/tree/main)
            """,
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
        "--inputs-folder",
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
        "--labels-folder",
        required=True,
        type=str,
        help="""Path to the folder containing the ground truth annotation
            maps. File names must match the case identifiers of the input
            images""",
    )
    parser.add_argument(
        "--output-folder",
        required=False,
        type=str,
        help="""[OPTIONAL] Path to the output folder. By default the only
            output is a 'metrics.json' file containing the evaluation results.
            If this flag is not included then nothing is saved.""",
    )
    parser.add_argument(
        "--save-probability-maps",
        required=False,
        action="store_true",
        help="""[OPTIONAL] Include this flag to save the predicted probability
            maps. Will be saved in their own folder in the output folder and
            named according to the case identifier""",
    )
    parser.add_argument(
        "--save-detection-maps",
        required=False,
        action="store_true",
        help="""[OPTIONAL] Include this flag to save the predicted lesion
            detection maps. Will be saved in their own folder in the output
            folder and named according to the case identifier""",
    )
    parser.add_argument(
        "--save-annotations",
        required=False,
        action="store_true",
        help="""[OPTIONAL] Include this flag to save the final predicted
            annotations/segmentations""",
    )

    args = parser.parse_args()

    pred_and_eval(
        config_path=args.config_path,
        inputs_folder=args.inputs_folder,
        labels_folder=args.labels_folder,
        output_folder=args.output_folder,
        save_probability_maps=args.save_probability_maps,
        save_detection_maps=args.save_detection_maps,
        save_annotations=args.save_annotations,
    )


if __name__ == "__main__":
    main()

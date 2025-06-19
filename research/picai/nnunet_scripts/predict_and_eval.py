import argparse
import sys
from logging import INFO
from os.path import exists, join

from flwr.common.logger import log

from research.picai.nnunet_scripts.eval import generate_detection_maps, get_picai_metrics
from research.picai.nnunet_scripts.predict import predict


def pred_and_eval(
    config_path: str,
    input_folder: str,
    label_folder: str,
    output_folder: str,
) -> None:
    """
    Runs prediction and evaluation pipeline for nnunet models. First runs
    inference for one or more models and nnunet configs. Converts the model
    output logits to probabilities, then ensembles the probabilities and
    computes the predicted annotations. It also derives detection maps from
    the ensembled probabilities and computes the official PICAI metrics from
    using the picai_eval API from the detection maps.

    Args:
        config_path (str): Path to a yaml config file. The three required keys
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
            and their corresponding values as specified by nnunet.
        input_folder (str): Path to the folder containing the raw input data
            that has not been processed by nnunet yet. File names must follow
            the nnunet convention where each channel modality is stored as a
            separate file. File names should be case-identifier_0000 where
            0000 is a 4 digit integer representing the channel/modality. All
            cases must have the same N channels numbered from 0 to N.
        label_folder (str): Path to the folder containing the ground truth
            annotation maps. File names must match the case identifiers of the
            input images
        output_folder (str): Path to the output folder. By default this script
            will save the predicted probabilities, detection maps and
            annotations.
    """
    det_map_folder_name = "detection_maps"
    probs_folder_name = "predicted_probability_maps"
    annotation_folder_name = "predicted_annotations"
    predict(
        config_path=config_path,
        input_folder=input_folder,
        output_folder=output_folder,
        probs_folder_name=probs_folder_name,
        annotations_folder_name=annotation_folder_name,
        verbose=True,
    )
    log(INFO, "")
    generate_detection_maps(
        input_folder=join(output_folder, probs_folder_name),
        output_folder=join(output_folder, det_map_folder_name),
        transforms=None,
        npz_key="probabilities",
        num_threads=None,  # Let threadpool determine optimal num threads
        postfixes=[""],
        extensions=[".npz"],  # Probability maps saved as npz files
        verbose=True,
    )
    log(INFO, "")
    metrics = get_picai_metrics(
        detection_map_folder=join(output_folder, det_map_folder_name),
        ground_truth_annotations_folder=label_folder,
        num_threads=None,  # Let threadpool autodetermine num threads
        verbose=True,
    )
    metrics.save_minimal(join(output_folder, "metrics.json"))


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="Predict and Evaluate nnUNet models or nnUNet Ensembles",
        description="""Runs inference on raw input data given a number of
            compatible nnunet models. Then extracts detection maps from
            those predictions and evaluates the model using the standard
            PICAI evaluation metrics (Average Precision, PICAI Score, Area
            Under Receiving Operator Characteristic).""",
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
        "--label-folder",
        required=True,
        type=str,
        help="""Path to the folder containing the ground truth annotation
            maps. File names must match the case identifiers of the input
            images""",
    )
    parser.add_argument(
        "--output-folder",
        required=True,
        type=str,
        help="""Path to the output folder. By default this script will save
        the predicted probabilities, detection maps and annotations.""",
    )

    args = parser.parse_args()

    output_folder = args.output_folder
    if exists(output_folder):
        print(f"Found existing folder for output folder : {output_folder}")
        print("If you choose to proceed existing files may be overwritten")
        while True:
            proceed = input("Do you wish to proceed (y/n)?: ")
            if proceed in {"y", "Y"}:
                break
            if proceed in {"n", "N"}:
                sys.exit()
            else:
                print("Did not get valid input. Please enter 'y' or 'n'")

    pred_and_eval(
        config_path=args.config_path,
        input_folder=args.input_folder,
        label_folder=args.label_folder,
        output_folder=args.output_folder,
    )


if __name__ == "__main__":
    main()

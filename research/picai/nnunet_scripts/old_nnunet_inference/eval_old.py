import argparse
import contextlib
import os
import warnings
from collections import defaultdict
from collections.abc import Iterable, Sequence
from os.path import join
from typing import Any

import numpy as np
from numpy.typing import NDArray


with contextlib.redirect_stdout(open(os.devnull, "w")):
    from picai_eval import evaluate, evaluate_folder  # type: ignore[import-not-found]
    from picai_eval.image_utils import read_image  # type: ignore[import-not-found]
    from picai_eval.metrics import Metrics as PicaiEvalMetrics  # type: ignore[import-not-found]
    from report_guided_annotation import extract_lesion_candidates  # type: ignore[import-not-found]

warnings.simplefilter("ignore", category=FutureWarning)


def load_images_from_folder(
    folder: str,
    case_identifiers: list[str],
    postfixes: list[str] | None = None,
    extensions: Sequence[str] = [".nii.gz", ".nii", ".mha", ".mhd", ".npz", ".npy"],
) -> NDArray:
    """
    Loads images from a folder given a list of case identifiers.

    Args:
        folder (str): The folder containing the images
        case_identifiers (list[str]): A list of case identifiers for each
            file. Typically just the filenames without the extension
        postfixes (list[str] | None, optional): A list of strings to append
            to the case identifiers when looking for files. For example
            '_labels'. Defaults to None.
        extensions (list[str], optional): A list of possible image extensions.
            Defaults to [".nii.gz", ".nii", ".mha", ".mhd", ".npz", ".npy"].

    Returns:
        (NDArray): A numpy array containing the images for each case identifier.
            The first dimension will be the number of images and those images
            will be in the same order as was given by the case_identifiers
            argument
    """
    if postfixes is None:
        postfixes = [""]

    images = []
    for case in case_identifiers:
        for postfix in postfixes:
            for ext in extensions:
                path = join(folder, case + postfix + ext)
                if os.path.exists(path):
                    images.append(read_image(path))

    return np.stack(images)


def get_detection_maps(probability_maps: NDArray) -> NDArray:
    """
    Generates detection maps from probability maps by doing lesion extraction.

    Args:
        probability_maps (NDArray): A numpy array containing the predicted
            probability maps. Should be shape (num_samples, num_classes, ...)

    Returns:
        (NDArray): A numpy array of the detection maps for each class. Note that
            a detection map is not created for the background class which is
            assumed to be at index 0. Therefore the output shape is
            (num_samples, num_classes-1, ...)
    """
    num_classes = probability_maps.shape[1]

    detection_maps = []
    for prob_map in probability_maps:
        cls_probs = []
        for cls in range(1, num_classes):  # Ignore background class at idx 0
            cls_probs.append(extract_lesion_candidates(prob_map[cls], num_lesions_to_extract=5)[0])
        detection_maps.append(np.stack(cls_probs))

    return np.stack(detection_maps)


def get_picai_metrics(
    detection_maps: NDArray,
    ground_truth_annotations: NDArray,
    case_identifiers: Iterable[str] | None = None,
    **kwargs: Any,
) -> PicaiEvalMetrics:
    """
    Computes the picai evaluation metrics provided the predicted lesion
    detection maps and the ground truth annotations. Extends picai_eval to
    allow multiclass evaluation.

    Args:
        detection_maps (NDArray): The predicted lesion detection maps. Must
            have shape (num_samples, num_lesion_classes, ...). The background
            class should not be included in the detection maps
        ground_truth_annotations (NDArray): The ground truth annotations. Must
            have shape (num_samples, num_classes or num_lesion_classes, ...).
            If num_classes is provided, the function will attempt to remove
            the background class from index 0 for you
        case_identifiers (Iterable[str] | None, optional): A list of case
            identifiers. If not provided the subjects will be identified by
            their index Defaults to None.
        **kwargs: Keyword arguments for the picai_eval.evaluate function

    Raises:
        KeyError: If you try to use the y_det, y_true, or subject_list keyword
            arguments

    Returns:
        (picai_eval.metrics.Metrics): A picai eval metrics object that has
            combined the results from all classes into a single object
    """
    if "y_det" in kwargs or "y_true" in kwargs or "subject_list" in kwargs:
        raise KeyError(
            """Got one of 'y_det', 'y_true' or 'subject_list' in keyword
            arguments. The arguments to get_picai_metrics are being passed to
            these keys"""
        )

    if detection_maps.shape[1] != ground_truth_annotations.shape[1]:
        # Assume background class has not been removed
        ground_truth_annotations = ground_truth_annotations[:, 1:]
        assert detection_maps.shape == ground_truth_annotations.shape, (
            "Got unexpected shapes for detection maps and ground truth annotations"
        )

    # Evaluation must be calculated separately for each class
    num_classes = detection_maps.shape[1]
    metrics: list[PicaiEvalMetrics] = []
    for cls in range(num_classes):
        metrics.append(
            evaluate(
                y_det=detection_maps[:, cls],
                y_true=ground_truth_annotations[:, cls],
                subject_list=case_identifiers,
                **kwargs,
            )
        )

    subject_list = metrics[0].subject_list
    assert isinstance(subject_list, list), "Got unexpected subject list from picai eval metrics object"
    lesion_results: dict[Any, list] = defaultdict(list)
    lesion_weights: dict[Any, list] = defaultdict(list)
    case_targets: dict[Any, int] = defaultdict(int)
    case_preds: dict[Any, float] = defaultdict(float)
    for s in subject_list:
        # Ignoring mypy errors here for now because i don't know how to get around them
        [lesion_results[s].extend(m.lesion_results[s]) for m in metrics]  # type: ignore
        [lesion_weights[s].extend(m.lesion_weight[s]) for m in metrics]  # type: ignore
        case_targets[s] = int(any(m.case_target[s] for m in metrics))  # type: ignore
        case_preds[s] = max([m.case_pred[s] for m in metrics])  # type: ignore

    return PicaiEvalMetrics(
        lesion_results=lesion_results,
        case_target=case_targets,
        case_pred=case_preds,
        lesion_weight=lesion_weights,
        case_weight=metrics[0].case_weight,
        subject_list=metrics[0].subject_list,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pred-path",
        type=str,
        required=True,
        help="Path to the folder containing the predicted probability maps"
        "Only uses maps whose filename matches a file in the ground truth folder",
    )
    parser.add_argument(
        "--gt-path",
        type=str,
        required=True,
        help="Path to the folder with the ground truth segmentation maps"
        "Only uses maps whose filename matches a file in the predictions folder",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=False,
        default="metrics.json",
        help="Where to save the metrics as a json. Eg. 'path/to/metrics.json'",
    )

    args = parser.parse_args()

    # Compute picai metrics
    # Warning, this method assumes binary segmentation (ie. only two classes)
    # It also assumes the background class is at index 0
    metrics: PicaiEvalMetrics = evaluate_folder(
        y_det_dir=args.pred_path,
        y_true_dir=args.gt_path,
        y_det_postprocess_func=lambda pred: extract_lesion_candidates(pred)[0],
    )

    # Print metrics
    print(metrics)
    print("PICAI Score: ", metrics.score)

    # Save metrics
    metrics.save_minimal(args.output_path)


if __name__ == "__main__":
    main()

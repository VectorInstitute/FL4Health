import argparse
import concurrent.futures
import contextlib
import os
import time
import warnings
from collections.abc import Callable, Hashable
from logging import INFO
from os.path import exists, join
from pathlib import Path
from typing import Any

import numpy as np
import SimpleITK as sitk
from flwr.common.logger import log
from numpy.typing import NDArray


with contextlib.redirect_stdout(open(os.devnull, "w")):
    from picai_eval.eval import evaluate_case  # type: ignore
    from picai_eval.metrics import Metrics as PicaiEvalMetrics  # type: ignore
    from report_guided_annotation import extract_lesion_candidates  # type: ignore

warnings.simplefilter("ignore", category=FutureWarning)

# logger = multiprocessing.log_to_stderr()
# logger.setLevel(multiprocessing.SUBDEBUG)


def read_image(path: Path | str, npz_key: str | None = None) -> NDArray:
    """
    Taken from picai eval. Had to change one line so that they wouldn't throw away additional channels. They were
    assuming binary segmentation. Also made it work for any npz file.

    Args:
        path (Path | str): Path to the image file
        npz_key (str | None): If the file type is .npz, then a key must be provided to access the numpy array from the
            NpzFile object
    """
    if isinstance(path, Path):
        path = path.as_posix()

    if ".npy" in path:
        return np.load(path)
    if ".nii" in path or ".mha" in path or "mhd" in path:
        return sitk.GetArrayFromImage(sitk.ReadImage(path))
    if ".npz" in path:
        # read the nnU-Net format
        data = np.load(path)
        assert npz_key is not None, "Path leads to a .npz file but a key was not provided"
        data = data[npz_key]
        assert isinstance(data, np.ndarray), f"Was expecting a numpy array and got {type(data)}"
        return data.astype("float32")
    raise ValueError(f"Unexpected file path. Supported file formats: .nii(.gz), .mha, .npy and .npz. Got: {path}.")


def scan_folder_for_cases(
    folder: str | Path, postfixes: list[str] | None = None, extensions: list[str] | None = None
) -> list[str]:
    if postfixes is None:
        postfixes = [""]
    if extensions is None:
        extensions = [".npz", ".npy", ".nii.gz", ".nii", ".mha", ".mhd"]

    file_endings = [f"{pf}{ext}" for pf in postfixes for ext in extensions]

    file_list = os.listdir(folder)
    case_list = []
    for file in file_list:
        for end in file_endings:
            if file.endswith(end):
                case_id = file.replace(end, "")
                if case_id in case_list:
                    log(INFO, f"Found multiple files for {case_id}")
                    continue
                case_list.append(case_id)
                break

    return case_list


def get_case_files(
    folder: str | Path,
    case_identifiers: list[str],
    postfixes: list[str] | None = None,
    extensions: list[str] | None = None,
    basename: bool = False,
) -> list[str]:
    if postfixes is None:
        postfixes = [""]
    if extensions is None:
        extensions = [".npz", ".npy", ".nii.gz", ".nii", ".mha", ".mhd"]

    file_endings = [f"{pf}{ext}" for pf in postfixes for ext in extensions]

    selected_files = []
    for case in case_identifiers:
        found_file = False
        for end in file_endings:
            if exists(join(folder, f"{case}{end}")):
                if basename:
                    selected_files.append(f"{case}{end}")
                else:
                    selected_files.append(join(folder, f"{case}{end}"))
                found_file = True
        assert found_file, f"Could not find file for case: {case}"

    return selected_files


def generate_detection_map(
    probability_map: NDArray | str | Path,
    save_path: str | Path,
    npz_key: str | None = "probabilities",
    transforms: list[Callable[[NDArray], NDArray]] | None = None,
    **kwargs: Any,
) -> None:
    """
    Generates a detection map from a probability map by doing lesion
    extraction. Supports multiclass probability maps by extracting a separate
    lesion detection map for each class/channel.

    Args:
        probability_map (NDArray | str | Path]): One hot encoded
            probability map for a single image. Should be shape
            (num_classes, ...). num_classes includes the background class. If
            the probability maps are .npz files then a npz_key should be
            provided
        save_path (str | Path): Path to save the detection map. Will be
            saved as a numpy compressed .npz file under key 'detection_map'.
            Detection map will have shape (num_classes - 1, ...) since a
            detection map is not computed for the background class.
        npz_key (str | None): If probability_map is a path to a .npz
            file then a key must be provided to access the numpy array from
            the NpzFile object. Defaults to 'probabilities'
        transforms (list[Callable[[NDArray], NDArray]] | None): A list of
            transform functions to apply to the probability map before passing
            it to the lesion extraction method. The functions will be applied
            in the order they are given. Can be used, for example, to one hot
            encode binary probability maps if they are not already one hot
            encoded.
        **kwargs (Any): Keyword arguments to the extract_lesion_candidates
            function from the report_guided_annotation API.
    """
    if transforms is None:
        transforms = []

    if isinstance(probability_map, (str, Path)):
        probability_map = read_image(probability_map, npz_key=npz_key)

    for transform in transforms:
        probability_map = transform(probability_map)

    num_classes = probability_map.shape[0]
    cls_probs = []
    for cls in range(1, num_classes):
        cls_probs.append(extract_lesion_candidates(probability_map[cls], **kwargs)[0])

    np.savez_compressed(save_path, detection_map=np.stack(cls_probs))


def generate_detection_maps(
    input_folder: str | Path,
    output_folder: str | Path,
    transforms: list[Callable[[NDArray], NDArray]] | None = None,
    npz_key: str | None = "probabilities",
    num_threads: int | None = None,
    postfixes: list[str] | None = None,
    extensions: list[str] | None = None,
    verbose: bool = True,
    **kwargs: Any,
) -> list[str]:
    """
    Extracts lesions from predicted probability maps and saves the predicted
    lesions as detection maps in the output folder.

    Note that this method supports multiclass segmentation but the input
    predicted probability maps must be one hot encoded.

    Args:
        input_folder (_type_): Path to the folder containing the predicted
            probability maps. Each probability map must be saved as a separate
            file where the files basename will be used ato derive the case
            identifier. The probability maps must be one hot encoded and have
            shape (num_classes, ...) where num_classes includes the background
            class. Ie. binary segmentation would have num_classes=2
        output_folder (_type_): Path to the folder in which to save the
            detection maps. The detection maps will be save as numpy NpzFiles
            (.npz) under the key 'detection_map'. The detection maps will also
            be one hot encoded and have shape (num_classes - 1, ...). The
            number of classes decreases by one since a detection map is not
            generated for the background class.
            Ie. (num_lesion_classes == num_classes - 1). Note that this method
            will overwrite existing files if they already exist
        transforms (list[Callable[[NDArray], NDArray]] | None): A list of
            transform functions to apply to the probability map before passing
            it to the lesion extraction method. The functions will be applied
            in the order they are given. Can be used, for example, to one hot
            encode binary probability maps if they are not already one hot
            encoded. Note that since this method uses multiprocessing, the
            transform functions must be defined in a top-level module.
            Basically they can't be defined in 'if __name__ == '__main__' but
            everything else seems to work.
            See: https://stackoverflow.com/questions/72766345
        npz_key: If the predicted probability maps in the input folder are
            saved as .npz files, then this is the key used to access the numpy
            array from the NpzFile object. Defaults to 'probabilities'
        num_threads (int, optional): The maximum number of threads to allow
            when extracting the detection maps. If left as None, the number of
            threads is automatically determined.
        postfixes (list[str] | None): File postfixes (endings after the
            unique identifier but before the extension). Detection maps will
            only be generated for files with one or more of the specified
            postfixes. Postfixes are omitted from the returned case
            identifiers. Defaults to [""].
        extensions (list[str], optional): File extensions to allow. Detection
            maps will only be generated for files with on of the specified
            file extensions. File extensions are omitted from the returned
            case identifiers. Defaults to [".npz", ".npy", ".nii.gz", ".nii",
            ".mha", ".mhd"].
        verbose (bool): Whether or not to print a log statement with
            extraction results. Defaults to True.
        **kwargs (Any): Keyword arguments to the extract_lesion_candidates
            function from the report_guided_annotation API.

    Returns:
        (list[str]): A list of unique case identifiers. The case identifiers are
            the file basenames of the chosen input probability map files
            stripped of the the specified postfixes and their extension.
    """
    t_start = time.time()
    # Get list of input and output files
    case_ids = scan_folder_for_cases(input_folder, postfixes, extensions)
    input_files = get_case_files(input_folder, case_ids, postfixes, extensions)
    output_files = [join(output_folder, case + ".npz") for case in case_ids]

    if not exists(output_folder):
        os.makedirs(output_folder)

    # Using threading over multiprocessing since task is I/O bound
    with concurrent.futures.ThreadPoolExecutor(num_threads) as pool:
        [
            pool.submit(
                generate_detection_map,
                probability_map=input_file,
                save_path=output_file,
                npz_key=npz_key,
                transforms=transforms,
                **kwargs,
            )
            for input_file, output_file in zip(input_files, output_files)
        ]

    if verbose:
        elapsed = time.time() - t_start
        log(INFO, f"Extracted {len(case_ids)} detection maps in {elapsed:.2f}s (~{elapsed / len(case_ids):.3f}s/case)")

    return case_ids


def one_hot_ndarray(input: NDArray, num_classes: int) -> NDArray:
    one_hot = (np.arange(num_classes) == input[..., None]).astype(int)
    return np.moveaxis(one_hot, -1, 0)  # Moves num_classes dim to the front


def evaluate_case_multichannel(
    detection_map: NDArray | str | Path, ground_truth: NDArray | str | Path, **kwargs: Any
) -> tuple[list[tuple[int, float, float]], float, float, str]:
    if isinstance(detection_map, (str, Path)):
        detection_map = read_image(detection_map, npz_key="detection_map")
    if isinstance(ground_truth, (str, Path)):
        ground_truth = read_image(ground_truth)

    # We assume detection map is one hot encoded because otherwise there is no
    # way to determine proper rank

    if detection_map.ndim != ground_truth.ndim:
        # One hot encode and remove background class from ground truth
        num_classes = detection_map.shape[0] + 1  # Add one for background class
        ground_truth = one_hot_ndarray(ground_truth, num_classes)[1:]

    assert detection_map.shape[0] == ground_truth.shape[0], (
        f"Was expecting detection map and ground truth to be the same shape detection_map: {detection_map.shape} "
        f"ground_truth: {ground_truth.shape}"
    )

    num_lesion_classes = detection_map.shape[0]  # One less than num_classes since it does not include background

    # The return type signature of evaluate_case is just wrong
    results = [
        evaluate_case(y_det=detection_map[channel], y_true=ground_truth[channel], **kwargs)
        for channel in range(num_lesion_classes)
    ]

    lesion_result = []
    case_conf: float = 0
    case_weight: float = kwargs["sample_weight"] if kwargs.get("sample_weight") is not None else 1
    case_id: str

    # The return signature for evaluate_case is just wrong.
    # So we have to ignore a mypy error here
    for lesions, conf, _, id in results:  # type: ignore
        lesion_result.extend(lesions)
        case_conf = max(case_conf, conf)
        case_id = id

    return lesion_result, case_conf, case_weight, case_id


def get_picai_metrics(
    detection_map_folder: str | Path,
    ground_truth_annotations_folder: str | Path,
    num_threads: int | None = None,
    sample_weights: list[float] | None = None,
    case_identifiers: list[str] | None = None,
    verbose: bool = True,
    **kwargs: Any,
) -> PicaiEvalMetrics:
    """
    Computes the picai evaluation metrics provided the predicted lesion detection maps and the ground truth
    annotations. Extends picai_eval to allow multiclass evaluation.

    Args:
        detection_map_folder (str | Path): Path to the folder containing the detection maps
        ground_truth_annotations_folder (NDArray): The ground truth annotations. Must have shape
            (num_samples, num_classes or num_lesion_classes, ...). If num_classes is provided, the function will
            attempt to remove the background class from index 0 for you
        num_threads (int | None): Number of threads to be used in the computations. Defaults to None.
        sample_weights: (list[float] | None): Weights on each sample to be used in metrics calculations. Defaults to
            None.
        case_identifiers (list[str] | None, optional): A list of case identifiers. If not provided the subjects will
            be identified by their index Defaults to None.
        verbose (bool): Whether or not to print a log statement summarizing results. Defaults to True
        **kwargs: Keyword arguments for the picai_eval.eval.evaluate_case function

    Raises:
        KeyError: If you try to use the y_det, y_true, or subject_list keyword
            arguments

    Returns:
        (picai_eval.metrics.Metrics): A picai eval metrics object that has
            combined the results from all classes into a single object
    """
    t_start = time.time()
    if "y_det" in kwargs or "y_true" in kwargs or "idx" in kwargs or "sample_weight" in kwargs:
        raise KeyError(
            f"""Got one or more of 'y_det', 'y_true', 'idx' or 'sample_weight'
            in keyword arguments {kwargs.keys()}.
            These keys are being used by get_picai_metrics. 'y_det' and
            'y_true' are derived from the detection maps and ground truth
            folders respectively. 'idx' is being used by the case_identifiers.
            If you have a list of case_id's use case_identifiers instead."""
        )

    # Get list of files,
    case_ids = scan_folder_for_cases(detection_map_folder) if case_identifiers is None else case_identifiers
    det_files = get_case_files(detection_map_folder, case_ids)
    gt_files = get_case_files(ground_truth_annotations_folder, case_ids)

    # Set default sample weights
    if sample_weights is None:
        sample_weights = [1] * len(case_ids)

    # Initialize variables to hold results
    case_targets: dict[Hashable, int] = {}
    case_weights: dict[Hashable, float] = {}
    case_preds: dict[Hashable, float] = {}
    lesion_results: dict[Hashable, list[tuple[int, float, float]]] = {}
    lesion_weights: dict[Hashable, list[float]] = {}

    # Evaluation must be calculated separately for each class
    with concurrent.futures.ThreadPoolExecutor(num_threads) as pool:
        futures = {
            pool.submit(evaluate_case_multichannel, detection_map=det_map, ground_truth=gt, idx=case, **kwargs): case
            for det_map, gt, case in zip(det_files, gt_files, case_ids)
        }

        for future in concurrent.futures.as_completed(futures):
            case = futures[future]
            lesion_results[case], case_preds[case], case_weights[case], _ = future.result()

            # Each lesion_results[case] is a list of tuples where each tuple
            # represents a predicted lesion. The tuple values are
            # (ground_truth_label, confidence, overlap)
            if len(lesion_results[case]):
                case_targets[case] = np.max([lr[0] for lr in lesion_results[case]])
            else:
                case_targets[case] = 0

            lesion_weights[case] = [case_weights[case]] * len(lesion_results[case])

        metrics = PicaiEvalMetrics(
            lesion_results=lesion_results,
            lesion_weight=lesion_weights,
            case_pred=case_preds,
            case_target=case_targets,
            case_weight=case_weights,
        )

        if verbose:
            elapsed = time.time() - t_start
            log(INFO, f"Computed metrics for {len(case_ids)} cases in {elapsed:.1f}s")
            log(INFO, f"\t{metrics}")
            log(INFO, f"\tPICAI Score: {metrics.score:.4f}")

        return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--probs-path",
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
        help="Folder in which to store the detection maps and the metrics json",
    )

    args = parser.parse_args()

    det_maps_path = join(args.output_path, "detection_maps")

    # Generate the detection maps
    t = time.time()
    case_ids = generate_detection_maps(args.probs_path, det_maps_path, extensions=[".npz"])
    elapsed = time.time() - t
    log(INFO, f"Extracted {len(case_ids)} detection maps in {elapsed:.2f}s (~{elapsed / len(case_ids):.3f}s/case)")

    metrics = get_picai_metrics(det_maps_path, args.gt_path)
    metrics.save(join(args.output_path, "metrics.json"))
    print(metrics)
    print(metrics.score)


if __name__ == "__main__":
    main()

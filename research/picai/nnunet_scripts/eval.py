import argparse
import warnings

from picai_eval import evaluate_folder
from picai_eval.metrics import Metrics
from report_guided_annotation import extract_lesion_candidates

warnings.simplefilter("ignore", category=FutureWarning)


def evaluate() -> None:
    pass


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
    metrics: Metrics = evaluate_folder(
        y_det_dir=args.pred_path,
        y_true_dir=args.gt_path,
        y_det_postprocess_func=lambda pred: extract_lesion_candidates(pred)[0],
    )

    # Print metrics
    print(metrics)
    print("PICAI Score: ", metrics.score)

    # Save metrics
    metrics.save(args.output_path)


if __name__ == "__main__":
    main()

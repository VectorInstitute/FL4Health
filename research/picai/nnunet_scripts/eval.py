import argparse

from picai_eval import evaluate_folder


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pred_path",
        type=str,
        required=True,
        help="Path to the folder containing the predicted segmentation maps"
        "Only uses maps whose filename matches a file in the ground truth folder",
    )
    parser.add_argument(
        "--gt_path",
        type=str,
        required=True,
        help="Path to the folder with the ground truth segmentation maps"
        "Only uses maps whose filename matches a file in the predictions folder",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=False,
        default="metrics.json",
        help="Where to save the metrics as a json. Eg. 'path/to/metrics.json'",
    )

    args = parser.parse_args()

    # Compute picai metrics
    metrics = evaluate_folder(
        y_det_dir=args.pred_path,
        y_true_dir=args.gt_path,
    )

    # Print metrics
    print("Metrics:")
    print(metrics)
    print("PICAI Score: ", metrics.score)

    # Save metrics
    metrics.save(args.output_path)


if __name__ == "__main__":
    main()

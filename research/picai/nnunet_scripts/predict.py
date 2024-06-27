import argparse

from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the results folder for the trained model. Must contain the"
        "dataset.json, the model plans json, and folders for at least one fold with the"
        "associated checkpoint files in them",
    )
    parser.add_argument(
        "--raw_inputs", type=str, required=True, help="Path to the raw (not yet processed by nnUNet) input images"
    )
    parser.add_argument(
        "--output_path", type=str, required=True, help="Where to store the predicted segmentation maps"
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        required=False,
        default="checkpoint_final.pth",
        help="Name of the checkpoint file to use when loading weights. If inference"
        " is run on multiple folds, the checkpoint name must be the same for all"
        "folds",
    )
    parser.add_argument(
        "--folds",
        type=str,
        required=False,
        help="The folds (0 to 4) to use when loading model checkpoints. Uses all"
        "5 folds (provided the trained model is available for those folds)"
        "by default. The prediction is an average of the loaded models."
        "Note that this flag can also be set to 'all' if there exists a model"
        "trained on all the training data",
    )

    args = parser.parse_args()

    # Load model checkpoint
    predictor = nnUNetPredictor()
    predictor.initialize_from_trained_model_folder(
        model_training_output_dir=args.model_path, checkpoint_name=args.ckpt, use_folds=args.folds
    )

    # Predict
    predictor.predict_from_files(
        list_of_lists_or_source_folder=args.raw_inputs,
        output_folder_or_list_of_truncated_output_files=args.output_path,
    )


if __name__ == "__main__":
    main()

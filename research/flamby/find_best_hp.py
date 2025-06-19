import argparse
import os
from logging import INFO

import numpy as np
from flwr.common.logger import log


def get_hp_folders(hp_sweep_dir: str) -> list[str]:
    paths_in_hp_sweep_dir = [os.path.join(hp_sweep_dir, contents) for contents in os.listdir(hp_sweep_dir)]
    return [hp_folder for hp_folder in paths_in_hp_sweep_dir if os.path.isdir(hp_folder)]


def get_run_folders(hp_dir: str) -> list[str]:
    run_folder_names = [folder_name for folder_name in os.listdir(hp_dir) if "Run" in folder_name]
    return [os.path.join(hp_dir, run_folder_name) for run_folder_name in run_folder_names]


def get_weighted_loss_from_server_log(
    run_folder_path: str, experiment_name: str, is_partial_efficient_net: bool
) -> float:
    server_log_path = os.path.join(run_folder_path, "server.out")
    with open(server_log_path, "r") as handle:
        files_lines = handle.readlines()
        if experiment_name in {"fed_heart_disease", "fed_ixi"}:
            line_to_convert = files_lines[-1].strip()
        elif is_partial_efficient_net:
            # FENDA, FedPer, PerFCL and MOON which use only feature extractor of Efficient Net,
            # doesn't log the same way as other Fed-ISIC methods
            line_to_convert = files_lines[-1].strip()
        else:
            line_to_convert = files_lines[-3].strip()
        return float(line_to_convert)


def main(hp_sweep_dir: str, experiment_name: str, is_partial_efficient_net: bool) -> None:
    hp_folders = get_hp_folders(hp_sweep_dir)
    best_avg_loss: float | None = None
    best_folder = ""
    for hp_folder in hp_folders:
        run_folders = get_run_folders(hp_folder)
        hp_losses = []
        for run_folder in run_folders:
            run_loss = get_weighted_loss_from_server_log(run_folder, experiment_name, is_partial_efficient_net)
            hp_losses.append(run_loss)
        current_avg_loss = float(np.mean(hp_losses))
        if best_avg_loss is None or current_avg_loss <= best_avg_loss:
            log(INFO, f"Current Loss: {current_avg_loss} is lower than Best Loss: {best_avg_loss}")
            log(INFO, f"Best Folder: {hp_folder}, Previous Best: {best_folder}")
            best_avg_loss = current_avg_loss
            best_folder = hp_folder
    log(INFO, f"Best Loss: {best_avg_loss}")
    log(INFO, f"Best Folder: {best_folder}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Holdout Global")
    parser.add_argument(
        "--hp_sweep_dir",
        action="store",
        type=str,
        help="Path to the artifacts of the hyper-parameter sweep script",
        required=True,
    )
    parser.add_argument(
        "--experiment_name",
        action="store",
        type=str,
        help="Name of the experiment type that is being analyzed. Options are: fed_isic, fed_heart_disease, fed_ixi",
        required=True,
    )
    parser.add_argument(
        "--is_partial_efficient_net",
        action="store_true",
        help="""boolean to indicate whether we are using the full efficient net or only its feature extractor such as:
        in the case of fenda, fedper, PerFCL, moon (only relevant for fed_isic experiments)""",
    )
    args = parser.parse_args()

    log(INFO, f"Hyperparameter Sweep Directory: {args.hp_sweep_dir}")
    log(INFO, f"Experiment Names: {args.experiment_name}")
    log(INFO, f"Is Partial Efficient Net: {args.is_partial_efficient_net}")
    main(args.hp_sweep_dir, args.experiment_name, args.is_partial_efficient_net)

import argparse
import os
from logging import INFO
from pathlib import Path

import torch
from data.data import load_train_delirium, load_train_mortality
from flwr.common.logger import log
from torch import nn

from fl4health.checkpointing.checkpointer import BestMetricTorchCheckpointer
from fl4health.metrics import AccumulationMeter, Metric
from research.gemini.delirium_models.NN import NN as delirium_model
from research.gemini.metrics.metrics import Accuracy, BinaryF1, BinaryRocAuc
from research.gemini.mortality_models.NN import NN as mortality_model


def main(
    data_path: Path,
    metrics: list[Metric],
    device: torch.device,
    hospitals_id: list[str],
    learning_task: str,
    batch_size: int,
    num_epochs: int,
    learning_rate: float,
    checkpoint_stub: str,
    run_name: str = "",
) -> None:
    # Create the model
    if learning_task == "mortality":
        model: nn.Module = mortality_model(input_dim=35, output_dim=1).to(device)
        # Load training and validation data from the given hospitals.
        train_loader, val_loader, num_examples = load_train_mortality(data_path, batch_size, hospitals_id)
    else:
        model: nn.Module = delirium_model(input_dim=8093, output_dim=1).to(device)
        train_loader, val_loader, num_examples = load_train_delirium(data_path, batch_size, hospitals_id)
    log(INFO, f"Client hospitals {hospitals_id}")

    # Checkpointing: create a string of the names of the hospitals
    hospital_names = ",".join(hospitals_id)
    checkpoint_dir = os.path.join(checkpoint_stub, run_name)
    checkpoint_name = f"client{hospital_names}_best_model.pkl"
    checkpointer = BestMetricTorchCheckpointer(checkpoint_dir, checkpoint_name, maximize=False)

    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # Perform training and validation
    train_meter = AccumulationMeter(metrics, "train_meter")
    val_meter = AccumulationMeter(metrics, "val_meter")

    for epoch in range(num_epochs):
        log(INFO, f"Epoch: {epoch}")

        train_meter.clear()
        training_loss_sum = 0.0

        # Training
        model.train()
        for input, target in train_loader:
            input, target = input.to(device), target.to(device)
            preds = model(input)
            train_loss = criterion(preds, target)
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            train_meter.update(preds, target)
            training_loss_sum += train_loss.item()

        # Evaluation
        val_meter.clear()
        validation_loss_sum = 0

        model.eval()
        with torch.no_grad():
            for input, target in val_loader:
                input, target = input.to(device), target.to(device)
                preds = model(input)
                val_loss = criterion(preds, target)

                val_meter.update(preds, target)
                validation_loss_sum += val_loss.item()

        epoch_val_loss = validation_loss_sum / len(val_loader)

        # Checkpointing the model based on validation loss
        checkpointer.maybe_checkpoint(model, epoch_val_loss)

    log(INFO, "Training finished")
    log(INFO, f"Best Loss seen by the client: \n{checkpointer.best_metric}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Local Training")
    parser.add_argument(
        "--task", action="store", type=str, default="mortality", help="GEMINI usecase: mortality or delirium"
    )
    parser.add_argument("--hospital_id", nargs="+", default=["THPC", "SMH"], help="ID of hospitals")
    parser.add_argument("--batch_size", action="store", type=int, default=64, help="Batch size")
    parser.add_argument("--num_epochs", action="store", type=int, default=100, help="Number of epochs")
    parser.add_argument(
        "--artifact_dir",
        action="store",
        type=str,
        help="Path to save client artifacts such as logs and model checkpoints",
        required=True,
    )
    parser.add_argument(
        "--run_name",
        action="store",
        help="Name of the run, model checkpoints will be saved under a subfolder with this name",
        required=True,
    )
    parser.add_argument(
        "--learning_rate", action="store", type=float, help="Learning rate for local optimization", default=0.01
    )
    args = parser.parse_args()

    if args.task == "mortality":
        data_path = Path("mortality_data")
    elif args.task == "delirium":
        data_path = Path("delirium_data")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    log(INFO, f"Device to be used: {device}")
    log(INFO, f"Task: {args.task}")

    main(
        data_path,
        [BinaryRocAuc(), BinaryF1(), Accuracy()],
        device,
        args.hospital_id,
        args.task,
        args.batch_size,
        args.num_epochs,
        args.learning_rate,
        args.artifact_dir,
        args.run_name,
    )

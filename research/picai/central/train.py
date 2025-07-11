import argparse

import torch
from torchmetrics.classification import MultilabelAveragePrecision

from fl4health.metrics import TorchMetric
from fl4health.metrics.metric_managers import MetricManager
from research.picai.data.data_utils import get_dataloader, get_img_and_seg_paths, get_img_transform, get_seg_transform
from research.picai.losses import FocalLoss
from research.picai.model_utils import get_model
from research.picai.single_node_trainer import SingleNodeTrainer


def main() -> None:
    # command line arguments for hyperparameters and I/O paths
    parser = argparse.ArgumentParser(description="Command Line Arguments for Training Script")

    # Data related arguments
    parser.add_argument(
        "--overviews_dir", type=str, help="Base path to training/validation data overviews", required=True
    )
    parser.add_argument("--fold", type=int, required=True, help="Which fold to perform experiment")
    parser.add_argument("--run_name", type=str, required=True, help="String used to name run")

    # Model related arguments
    parser.add_argument("--num_channels", type=int, default=3, help="Number of input channels/sequences")
    parser.add_argument("--num_classes", type=int, default=2, help="Number of classes at train-time")

    # training hyperparameters
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--checkpoint_dir", type=str, required=True, help="Path to save model checkpoints")
    parser.add_argument("--batch_size", type=int, default=4, help="Mini-batch size")

    # Dataloading arguments
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers to use in dataloader")

    args = parser.parse_args()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Initialize dataloaders, model, loss and optimizer to create trainer
    train_img_paths, train_seg_paths, train_class_proportions = get_img_and_seg_paths(
        args.overviews_dir, fold_id=args.fold, train=True
    )
    train_loader = get_dataloader(
        train_img_paths,
        train_seg_paths,
        args.batch_size,
        get_img_transform(),
        get_seg_transform(),
        num_workers=args.num_workers,
    )
    val_img_paths, val_seg_paths, _ = get_img_and_seg_paths(args.overviews_dir, fold_id=args.fold, train=False)
    val_loader = get_dataloader(
        val_img_paths,
        val_seg_paths,
        args.batch_size,
        get_img_transform(),
        get_seg_transform(),
        num_workers=args.num_workers,
    )

    model = get_model(device=device, spatial_dims=3, in_channels=args.num_channels, out_channels=args.num_classes)

    criterion = FocalLoss(alpha=train_class_proportions[-1].item())
    optimizer = torch.optim.Adam(params=model.parameters(), amsgrad=True)

    trainer = SingleNodeTrainer(
        train_loader=train_loader,
        val_loader=val_loader,
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        checkpoint_stub=args.checkpoint_dir,
        run_name=args.run_name,
    )

    # Define train and validation metrics and corresponding managers
    metrics = [
        TorchMetric(
            name="MLAP", metric=MultilabelAveragePrecision(average="macro", num_labels=2, thresholds=3).to(device)
        )
    ]
    train_metric_manager = MetricManager(metrics, "train")
    val_metric_manager = MetricManager(metrics, "val")

    trainer.train_by_epochs(args.num_epochs, train_metric_manager, val_metric_manager)


if __name__ == "__main__":
    main()

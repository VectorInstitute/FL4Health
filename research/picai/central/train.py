import argparse

import torch
from torchmetrics.classification import MultilabelAveragePrecision, MultilabelAccuracy
from fl4health.utils.metrics import MetricManager, TorchMetric
from research.picai.losses import FocalLoss
from research.picai.model_utils import get_model
from research.picai.data_utils import get_dataloader, get_img_and_seg_paths, get_img_transform, get_seg_transform
from research.picai.single_node_trainer import SingleNodeTrainer


def main():
    # command line arguments for hyperparameters and I/O paths
    parser = argparse.ArgumentParser(description='Command Line Arguments for Training Script')

    # Data related arguments
    parser.add_argument('--base_dir', type=str, default='/ssd003/projects/aieng/public/PICAI/',
                        help="Path to PICAI dataset")
    parser.add_argument('--overviews_dir', type=str,
                        default='/ssd003/projects/aieng/public/PICAI/workdir/results/UNet/overviews/Task2203_picai_baseline/',
                        help="Base path to training/validation data sheets")
    parser.add_argument('--fold', type=int, required=True, help="Which fold to perform experiment")

    # Model related arguments
    parser.add_argument('--num_channels', type=int, default=3, help="Number of input channels/sequences")
    parser.add_argument('--num_classes', type=int, default=2, help="Number of classes at train-time")

    # training hyperparameters
    parser.add_argument('--num_epochs', type=int, default=5, help="Number of training epochs")
    parser.add_argument('--checkpoint_dir', type=str, required=True, help="Path to save model checkpoints")
    parser.add_argument('--batch_size', type=int, default=8, help="Mini-batch size")

    args = parser.parse_args()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Initialize dataloaders, model, loss and optimizer to creater trainer
    train_img_paths, train_seg_paths, train_class_proportions = get_img_and_seg_paths(
        args.overviews_dir, args.base_dir, fold_id=args.fold, train=True)
    train_loader = get_dataloader(train_img_paths, train_seg_paths, args.batch_size,
                                  get_img_transform(), get_seg_transform(), num_workers=1)
    val_img_paths, val_seg_paths, _ = get_img_and_seg_paths(
        args.overviews_dir, args.base_dir, fold_id=args.fold, train=False)
    val_loader = get_dataloader(val_img_paths, val_seg_paths, args.batch_size,
                                get_img_transform(), get_seg_transform(), num_workers=1)

    model = get_model(
        device=device,
        spatial_dims=3,
        in_channels=args.num_channels,
        out_channels=args.num_classes
    )

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
        run_name="test_run",
    )

    # Define train and validation metrics and corresponding managers
    metrics = [TorchMetric(name="MLAP", metric=MultilabelAveragePrecision(
        average="macro", num_labels=2, thresholds=3).to(device))]
    train_metric_manager = MetricManager(metrics, "train")
    val_metric_manager = MetricManager(metrics, "val")

    trainer.train_by_epochs(args.num_epochs, train_metric_manager, val_metric_manager)


if __name__ == '__main__':
    main()
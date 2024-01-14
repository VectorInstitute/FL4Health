import argparse
import ast

import torch
from research.picai.losses import FocalLoss
from research.picai.train_utils import train, validate
from research.picai.model_utils import get_model
from research.picai.data_utils import get_dataloaders


def main():
    # command line arguments for hyperparameters and I/O paths
    parser = argparse.ArgumentParser(description='Command Line Arguments for Training Script')

    # data I/0 + experimental setup
    parser.add_argument('--base_dir', type=str, default='/ssd003/projects/aieng/public/PICAI/')
    parser.add_argument('--weights_dir', type=str, required=True,
                        help="Path to export model checkpoints")
    parser.add_argument('--overviews_dir', type=str, required=True,
                        help="Base path to training/validation data sheets")
    parser.add_argument('--folds', type=int, nargs='+', required=True,
                        help="Folds selected for training/validation run")

    # training hyperparameters
    parser.add_argument('--image_shape', type=int, nargs="+", default=[20, 256, 256],
                        help="Input image shape (z, y, x)")
    parser.add_argument('--num_channels', type=int, default=3,
                        help="Number of input channels/sequences")
    parser.add_argument('--num_threads', type=int, default=3,
                        help="Number of threads for dataloading")
    parser.add_argument('--num_classes', type=int, default=2,
                        help="Number of classes at train-time")
    parser.add_argument('--num_epochs', type=int, default=100,
                        help="Number of training epochs")

    # neural network-specific hyperparameters

    parser.add_argument('--model_type', type=str, default='unet',
                        help="string representation of model architecture")
    parser.add_argument('--channels_per_layer', type=str, default='[32, 64, 128, 256, 512, 1024]',
                        help="Neural network: number of encoder channels (as string representation)")
    parser.add_argument('--strides', type=str, default='[(2, 2, 2), (1, 2, 2), (1, 2, 2), (1, 2, 2), (2, 2, 2)]',
                        help="Neural network: convolutional strides (as string representation)")
    parser.add_argument('--batch_size', type=int, default=8,
                        help="Mini-batch size")

    args = parser.parse_args()
    args.strides = ast.literal_eval(args.strides)
    args.channels_per_layer = ast.literal_eval(args.channels_per_layer)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # for each fold
    for f in args.folds:

        # derive dataLoaders
        train_loader, val_loader, class_proportions = get_dataloaders(
            overview_dir=args.overviews_dir,
            base_dir=args.base_dir,
            batch_size=args.batch_size,
            num_threads=args.num_threads,
            fold_id=f
        )

        # model definition
        model = get_model(
            model_type=args.model_type,
            spatial_dims=len(args.image_shape),
            in_channels=args.num_channels,
            out_channels=args.num_classes,
            strides=args.strides,
            channels=args.channels_per_layer,
            device=device
        )
        criterion = FocalLoss(alpha=class_proportions[-1])
        optimizer = torch.optim.Adam(params=model.parameters(), amsgrad=True)
        train(model, train_loader, criterion, optimizer, device)
        validate(model, val_loader, criterion, device)


if __name__ == '__main__':
    main()

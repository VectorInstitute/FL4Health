
from research.picai.unet import UNet

def get_unet(args, device):
    """Select neural network architecture for given run"""

    if args.model_type == 'unet':
        model = UNet(
            spatial_dims=len(args.image_shape),
            in_channels=args.num_channels,
            out_channels=args.num_classes,
            strides=args.model_strides,
            channels=args.model_features
        )
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")

    model = model.to(device)
    print("Loaded Neural Network Arch.:", args.model_type)
    return model

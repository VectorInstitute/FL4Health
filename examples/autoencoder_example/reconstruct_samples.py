import argparse
import matplotlib.pyplot as plt
import torch
import os
import torchvision.transforms as transforms
from fl4health.utils.dataset import BaseDataset, MNISTDataset
from torch.utils.data import DataLoader

def reconstruct_save(experiment_name: str, laten_dim:int, target:int, dataset_path:str):
    checkpoint_path= f"examples/autoencoder_example/{experiment_name}/"
    model_checkpoint_path = os.path.join(checkpoint_path, "best_VAE_model.pkl")
    image_saving_path = os.path.join(checkpoint_path, f"reconstructed_image_{target}.png")
    # Load the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    autoencoder = torch.load(model_checkpoint_path)
    autoencoder.eval()

    # Load the original image (MNIST validation set)
    val_ds: BaseDataset = MNISTDataset(dataset_path, train=False, transform=transforms.ToTensor())
    validation_loader = DataLoader(val_ds, batch_size=1)
    targer_found = False
    while not targer_found:
        for image, label in validation_loader:
            label = label.item()  # Convert the label to an integer
            if label == target:
                original_image = image
                targer_found = True
    # Pass the original image through the autoencoder to get the reconstructed image
    reconstructed_image, _, _ = autoencoder(original_image)
    reconstructed_image = reconstructed_image.squeeze().detach().numpy() 
    original_image = original_image.squeeze().detach().numpy() 
    # Create a side-by-side comparison plot
    plt.subplot(1, 2, 1)
    plt.imshow(original_image, cmap='gray')
    plt.title("Original Image")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(reconstructed_image, cmap='gray')  # Display as grayscale
    plt.title("Reconstructed Image")
    plt.axis('off')
    plt.savefig(image_saving_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reconstruct samples")
    parser.add_argument(
        "--experiment_name",
        type=str,
        action="store",
        help="Name of the experiment to read the model from.",
        default="beta=1"
        )
    parser.add_argument(
        "--laten_dim",
        type=int,
        action="store",
        help="Dimention of the laten space used in the model",
        default=64
        )
    parser.add_argument(
        "--label",
        type=int,
        action="store",
        help="The output class that you want to reconstruct.",
        default=1
        )
    parser.add_argument(
        "--dataset_path",
        type=str,
        action="store",
        help="The path to the dataset used for reconstruction.",
        default="examples/datasets/MNIST"
        )
    args = parser.parse_args()
    assert 0<=args.label<10
    reconstruct_save(args.experiment_name, args.laten_dim, args.label, args.dataset_path)
    
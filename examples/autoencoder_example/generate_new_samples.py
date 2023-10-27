import argparse
import torch
import os
from typing import Optional
import matplotlib.pyplot as plt
from torchvision.utils import save_image

def generate_save(experiment_name:str, latent_dim:int, n_images:int, label: Optional[int]=None):
    checkpoint_path= f"examples/autoencoder_example/{experiment_name}/"
    if label!=None:
        model_checkpoint_path = os.path.join(checkpoint_path, "best_CVAE_model.pkl")
        image_saving_path = os.path.join(checkpoint_path, f"generated_image_{label}.png")
    else: 
        model_checkpoint_path = os.path.join(checkpoint_path, "best_VAE_model.pkl")
        image_saving_path = os.path.join(checkpoint_path, "generated_image.png")
    # Load the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    autoencoder = torch.load(model_checkpoint_path)
    autoencoder.eval()
    with torch.no_grad():
        # Sample from the latent space (from a unit Gaussian distribution)
        latent_samples = torch.randn(n_images, latent_dim).to(device)
        # Decode the latent vector to generate a new data sample
        if label!=None:
            # TODO: add a utils function for this
            zeros_t = torch.zeros(n_images, 10).to(device)
            y_tensor = torch.tensor(label, dtype=torch.int64)
            y_index = y_tensor.repeat(n_images,1)
            label = zeros_t.scatter_(1, y_index, 1)
            generated_sample = autoencoder.decoder(latent_samples, label).to(device)
        else: generated_sample = autoencoder.decode(latent_samples).to(device)
    if n_images>1:
        # For saving a batch of generated images we use torchvision.utils.save_image
        save_image(generated_sample.view(n_images, 1, 28, 28), image_saving_path)
    else:
        # To save a large single sample, we use matplotlib directly
        generated_sample = generated_sample.squeeze().detach().numpy() 
        plt.imshow(generated_sample, cmap='gray')
        plt.title("Generated Image")
        plt.axis('off')
        plt.savefig(image_saving_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate new samples")
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
        "--n_images",
        type=int,
        action="store",
        help="Number of the images to be created",
        default=64
        )
    parser.add_argument(
        "--label",
        type=int,
        action="store",
        help="Generate samples from a specific class [0-9]",
        default=None
        )
    args = parser.parse_args()
    generate_save(args.experiment_name, args.laten_dim, args.n_images, args.label)
    
import argparse
import torch
import os
import matplotlib.pyplot as plt
from torchvision.utils import save_image

def generate_save(experiment_name:str, latent_dim:int, n_images:int):
    checkpoint_path= f"examples/autoencoder_example/{experiment_name}/"
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
        generated_sample = autoencoder.decode(latent_samples).to(device)
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
    args = parser.parse_args()
    generate_save(args.experiment_name, args.laten_dim, args.n_images)
    
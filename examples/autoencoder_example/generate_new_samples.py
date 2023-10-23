import argparse
import torch
import os
import numpy as np
import matplotlib.pyplot as plt

def generate_save(experiment_name:str, latent_dim:int):
    checkpoint_path= f"examples/autoencoder_example/{experiment_name}/"
    model_checkpoint_path = os.path.join(checkpoint_path, "best_VAE_model.pkl")
    image_saving_path = os.path.join(checkpoint_path, "generated_image.png")

    # Load the model
    autoencoder = torch.load(model_checkpoint_path)
    autoencoder.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Sample from the latent space (from a unit Gaussian distribution)
    latent_sample = torch.randn(1, latent_dim).to(device)

    # Decode the latent vector to generate a new data sample
    with torch.no_grad():
        generated_sample = autoencoder.decode(latent_sample)
    generated_sample = generated_sample.squeeze().detach().numpy() 

    plt.imshow(generated_sample, cmap='gray')  # Display as grayscale
    plt.title("Generated Image")
    plt.axis('off')
    plt.savefig(image_saving_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot the clients' distributions")
    parser.add_argument(
        "--experiment_name",
        type=str,
        action="store",
        help="Name of the experiment.",
        default="beta=1"
        )
    parser.add_argument(
        "--laten_dim",
        type=int,
        action="store",
        help="Dimention of the laten space used in the model",
        default=64
        )
    args = parser.parse_args()
    generate_save(args.experiment_name, args.laten_dim)
    
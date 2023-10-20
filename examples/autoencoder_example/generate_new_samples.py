import torch
import os
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

checkpoint_path= "examples/autoencoder_example"
model_checkpoint_path = os.path.join(checkpoint_path, "best_VAE_model.pkl")
autoencoder = torch.load(model_checkpoint_path)
latent_dim = 16
autoencoder.eval()

# Set the device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Sample from the latent space (from a unit Gaussian distribution)
latent_sample = torch.randn(1, 16).to(device)

# Decode the latent vector to generate a new data sample
with torch.no_grad():
    generated_sample = autoencoder.decode(latent_sample)

generated_sample = generated_sample.squeeze().detach().numpy() 


plt.imshow(generated_sample, cmap='gray')  # Display as grayscale
plt.title("Generated Image")
plt.axis('off')
plt.savefig("examples/autoencoder_example/output/g_out.png")


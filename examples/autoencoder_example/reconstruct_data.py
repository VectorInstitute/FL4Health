"""
This file tests the autoencoder by feeding samples randomly taken
from each class and regenerating them to provide 
an illustration of the trained model outputs.
"""

import matplotlib.pyplot as plt
import torchvision
import torch
from pathlib import Path
import os
import torchvision.transforms as transforms
from fl4health.utils.dataset import BaseDataset, MNISTDataset
from torch.utils.data import DataLoader

checkpoint_path= "examples/autoencoder_example"
model_checkpoint_path = os.path.join(checkpoint_path, "best_VAE_model.pkl")
autoencoder = torch.load(model_checkpoint_path)
autoencoder.eval()

# Load the original image (example, replace with your image)
dataset_path = Path("examples/datasets/MNIST")
transform = transforms.Compose([transforms.ToTensor()])
val_ds: BaseDataset = MNISTDataset(dataset_path, train=False, transform=transform)
validation_loader = DataLoader(val_ds, batch_size=1)

# Define a dictionary to keep track of whether we have encountered a sample for each class
classes = [1,2,4]
class_sample_map = {}
for group in range(0,10): class_sample_map[group] = True if group in classes else False

# Iterate through the validation data loader
for image, label in validation_loader:
    label = label.item()  # Convert the label to an integer

    # Check if you have already encountered a sample for this class
    if class_sample_map[label] is True:

        # If not, store this sample and display it (e.g., using plt.imshow)
        class_sample_map[label] = image
        # Pass the original image through the autoencoder to get the reconstructed image
        reconstructed_image, _, _ = autoencoder(image)
        reconstructed_image = reconstructed_image.squeeze().detach().numpy() 
        image= image.squeeze().detach().numpy() 
        # Create a side-by-side comparison plot
        plt.subplot(1, 2, 1)
        plt.imshow(image, cmap='gray')
        plt.title("Original Image")
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(reconstructed_image, cmap='gray')  # Display as grayscale
        plt.title("Generated Image")
        plt.axis('off')
        plt.savefig(f"examples/autoencoder_example/output/{label}_out.png")


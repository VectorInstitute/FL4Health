"""
This file tests the autoencoder by feeding samples randomly taken
from each class and regenerating them to provide 
an illustration of the trained model outputs.
"""

import matplotlib.pyplot as plt
import torchvision
import torch
import os
from fl4health.utils.load_data import load_mnist_data

checkpoint_path= "examples/autoencoder_example"
model_checkpoint_path = os.path.join(checkpoint_path, "best_model.pkl")
autoencoder = torch.load(model_checkpoint_path)

# Load the original image (example, replace with your image)
dataset_path = "examples/datasets/"
_, val_loader, _ = load_mnist_data(dataset_path, batch_size=1)
# Define a dictionary to keep track of whether we have encountered a sample for each class
classes = 5
class_sample_map = {}
for group in range(0,classes):
    class_sample_map[group] = None

print(class_sample_map)
# Iterate through the validation data loader
for image, label in val_loader:
    label = label.item()  # Convert the label to an integer
    
    # Check if you have already encountered a sample for this class
    if label < classes and class_sample_map[label] is None:
        # If not, store this sample and display it (e.g., using plt.imshow)
        class_sample_map[label] = image
        # Pass the original image through the autoencoder to get the reconstructed image
        reconstructed_image = autoencoder(image)
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


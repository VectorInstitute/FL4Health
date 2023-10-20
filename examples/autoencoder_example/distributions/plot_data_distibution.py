import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Set the address of the saved data distribution pickles for each client
path_1 = "examples/autoencoder_example/distributions/client_1/distribution.pkl"
path_2 = "examples/autoencoder_example/distributions/client_2/distribution.pkl"
saving_dir = "examples/autoencoder_example/distributions/data_distribution.png"

with open(path_1, 'rb') as file:
    distribution_dict = pickle.load(file)
file.close()
# Load the label and count values and sort based on the label [1,2,..,10]
labels, counts = zip(*sorted(distribution_dict.items()))

# Create the subplot for the first client
plt.subplot(1, 2, 1)
plt.bar(labels, counts)
plt.xlabel('Class')
plt.ylabel('Count')
plt.title('Client 0 data distribution')
x_ticks = np.arange(0, 10, 1)
x_tick_labels = [str(tick) for tick in x_ticks] 
plt.xticks(x_ticks, x_tick_labels, rotation=90)


# Repeat the process for the second client
with open(path_2, 'rb') as file:
    distribution_dict = pickle.load(file)
file.close()
labels, counts = zip(*sorted(distribution_dict.items()))

plt.subplot(1, 2, 2)
plt.bar(labels, counts)
plt.xlabel('Class')
plt.ylabel('Count')
plt.title('Client 1 data distribution')
plt.xticks(x_ticks, x_tick_labels, rotation=90)

plt.savefig(saving_dir)
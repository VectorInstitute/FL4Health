import argparse
import pickle
import matplotlib.pyplot as plt
import numpy as np

def plot_distribution(experiment_name:str, client_num:int):
    saving_dir = f"examples/autoencoder_example/{experiment_name}/data_distribution.png"

    for client_id in range(client_num):
        distribution_path = f"examples/autoencoder_example/{experiment_name}/client_{client_id}/distribution.pkl"
        with open(distribution_path, 'rb') as file:
            distribution_dict = pickle.load(file)
        file.close()
        # Load the label and count values and sort based on the label [1,2,..,10]
        labels, counts = zip(*sorted(distribution_dict.items()))
        # Create the subplot for the client
        plt.subplot(1, client_num, client_id+1)
        plt.bar(labels, counts)
        plt.xlabel('Class')
        plt.ylabel('Count')
        plt.title(f"Client_{client_id} data distribution")
        x_ticks = np.arange(0, 10, 1)
        x_tick_labels = [str(tick) for tick in x_ticks] 
        plt.xticks(x_ticks, x_tick_labels, rotation=90)

    plt.savefig(saving_dir)




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
        "--n_clients",
        type=int,
        help="Number of the clients in this experiment.",
        default=2,
    )
    args = parser.parse_args()
    plot_distribution(args.experiment_name, args.n_clients)
    
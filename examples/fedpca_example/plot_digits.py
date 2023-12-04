import matplotlib.pyplot as plt
import numpy as np
from flwr.common.typing import NDArray


def plot_digits(X: NDArray, title: str) -> None:
    """Small helper function to plot some digits."""
    fig, axs = plt.subplots(nrows=4, ncols=4, figsize=(8, 8))
    for img, ax in zip(X, axs.ravel()):
        ax.imshow(img.reshape((16, 16)), cmap="Greys")
        ax.axis("off")
    fig.suptitle(title, fontsize=24)
    plt.show()


if __name__ == "__main__":
    with open("pcs.npy", "rb") as f:
        principal_components = np.load(f)

    print(principal_components.shape)
    plot_digits(np.squeeze(principal_components), title="merged pcs")

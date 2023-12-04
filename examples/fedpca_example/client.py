import argparse
from pathlib import Path
from random import randrange

import flwr as fl
from flwr.common import NDArray
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import MinMaxScaler

from fl4health.clients.fed_pca_client import FedPCAClient


class MNISTFedPCAClient(FedPCAClient):
    def __init__(self, data_path: Path, pc_path: Path, n_components: int) -> None:
        super().__init__(data_path, pc_path, n_components)

    def get_data_numpy(self) -> NDArray:
        X, y = fetch_openml(data_id=41082, as_frame=False, return_X_y=True, parser="pandas")
        X = MinMaxScaler().fit_transform(X)
        digit_label = str(randrange(1, 11))
        digits_indices = y == digit_label
        return X[digits_indices]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FL Client Main")
    parser.add_argument("--dataset_path", action="store", type=str, help="Path to the local dataset", default="./")
    parser.add_argument("--n_components", action="store", type=int, help="Number of principal components")
    parser.add_argument("--pc_path", action="store", type=str, help="Path to the local dataset", default="pcs.npy")
    args = parser.parse_args()

    data_path = Path(args.dataset_path)
    pc_path = Path(args.pc_path)
    client = MNISTFedPCAClient(data_path, pc_path, n_components=args.n_components)
    fl.client.start_numpy_client(server_address="0.0.0.0:8080", client=client)

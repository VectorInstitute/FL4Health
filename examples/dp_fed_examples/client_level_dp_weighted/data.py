import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset


class Scaler:
    def __init__(self) -> None:
        self.scaler = MinMaxScaler()

    def __call__(self, train_x: np.ndarray, val_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        scaled_train_x = self.scaler.fit_transform(train_x)
        scaled_val_x = self.scaler.transform(val_x)
        return scaled_train_x, scaled_val_x


def load_data(data_dir: Path, batch_size: int, scaler_bytes: bytes) -> tuple[DataLoader, DataLoader, dict[str, int]]:
    data = pd.read_csv(data_dir, index_col=False)
    features = data.loc[:, data.columns != "label"].values
    labels = data["label"].values
    n_samples = data.shape[0]

    scaler = pickle.loads(scaler_bytes)
    train_samples = int(n_samples * 0.8)
    train_features, train_labels = features[:train_samples, :], labels[:train_samples]
    val_features, val_labels = features[train_samples:, :], labels[train_samples:]
    train_features, val_features = scaler(train_features, val_features)
    train_x, train_y = torch.from_numpy(train_features).float(), torch.from_numpy(train_labels).float()
    val_x, val_y = torch.from_numpy(val_features).float(), torch.from_numpy(val_labels).float()
    train_ds, val_ds = TensorDataset(train_x, train_y), TensorDataset(val_x, val_y)
    train_loader = DataLoader(train_ds, batch_size=batch_size)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    num_examples = {"train_set": train_samples, "validation_set": n_samples - train_samples}

    return train_loader, val_loader, num_examples

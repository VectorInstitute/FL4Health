import argparse
import os
import pickle
from collections.abc import Hashable
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from PIL import Image
from torchvision.transforms import ToTensor


def filter_and_save_data(metadata: pd.DataFrame, top_sirna_ids: list[int], cell_type: str, output_path: Path) -> None:
    """
    Filters data for the given cell type and frequency of their ``sirna_id`` and saves it to a CSV file.

    Args:
        metadata (pd.DataFrame): Metadata containing information about all images.
        top_sirna_ids (list[int]): Top ``sirna_id`` values to filter by.
        cell_type (str): Cell type to filter by.
        output_path (Path): Path to save the filtered metadata.
    """
    filtered_metadata = metadata[(metadata["sirna_id"].isin(top_sirna_ids)) & (metadata["cell_type"] == cell_type)]
    filtered_metadata.to_csv(output_path, index=False)


def load_image(row: dict[Hashable, Any], root: Path) -> torch.Tensor:
    """
    Load an image tensor for a given row of metadata.

    Args:
        row (dict[str, Any]): A row of metadata containing experiment, plate, well, and site information.
        root (Path): Root directory containing the image files.

    Returns:
        (torch.Tensor): The loaded image tensor.
    """
    experiment = row["experiment"]
    plate = row["plate"]
    well = row["well"]
    site = row["site"]

    images = []
    # Rxrx1 originally consists of 6 channels, but to reduce the computational cost, we only use 3 channels
    # following previous works such as https://github.com/p-lambda/wildYe.
    for channel in range(1, 4):
        image_path = os.path.join(root, f"images/{experiment}/Plate{plate}/{well}_s{site}_w{channel}.png")
        if not Path(image_path).exists():
            raise FileNotFoundError(f"Image not found at {image_path}")
        image = ToTensor()(Image.open(image_path).convert("L"))
        images.append(image)

    # Concatenate the three channels into one tensor
    return torch.cat(images, dim=0)


def process_data(metadata: pd.DataFrame, input_dir: Path, output_dir: Path, client_num: int, type_data: str) -> None:
    """
    Process the entire dataset, loading image tensors for each row.

    Args:
        metadata (pd.DataFrame): Metadata containing information about all images.
        input_dir (Path): Input directory containing the image files.
        output_dir (Path): Output directory containing the image files.
        client_num (int): Client number to load data for.
        type_data (str): "train" or "test" to specify dataset type.
    """
    for i, row in metadata.iterrows():
        image_tensor = load_image(row.to_dict(), Path(input_dir))
        save_to_pkl(image_tensor, os.path.join(output_dir, f"{type_data}_data_{client_num + 1}", f"image_{i}.pkl"))


def save_to_pkl(data: torch.Tensor, output_path: str) -> None:
    """
    Save data to a pickle file.

    Args:
        data (torch.Tensor): Data to save.
        output_path (str): Path to the output pickle file.
    """
    with open(output_path, "wb") as f:
        pickle.dump(data, f)


def main(dataset_dir: Path) -> None:
    metadata_file = os.path.join(dataset_dir, "metadata.csv")
    output_dir = os.path.join(dataset_dir, "clients")

    os.makedirs(output_dir, exist_ok=True)

    data = pd.read_csv(metadata_file)

    # Get the top 50 `sirna_id`s by frequency
    top_sirna_ids = data["sirna_id"].value_counts().head(50).index.tolist()

    # Define cell types to distribute data based on them for each client
    cell_types = ["RPE", "HUVEC", "HEPG2", "U2OS"]
    output_files = [os.path.join(output_dir, f"meta_data_{i + 1}.csv") for i in range(len(cell_types))]

    # Filter and save data for each client
    for cell_type, output_path in zip(cell_types, output_files):
        filter_and_save_data(data, top_sirna_ids, cell_type, Path(output_path))

    for i, metadata_path in enumerate(output_files):
        metadata = pd.read_csv(metadata_path)

        # Split the metadata into train and test datasets
        train_metadata = metadata[metadata["dataset"] == "train"]
        test_metadata = metadata[metadata["dataset"] == "test"]

        process_data(train_metadata, dataset_dir, Path(output_dir), i, "train")
        process_data(test_metadata, dataset_dir, Path(output_dir), i, "test")


if __name__ == "__main__":
    # Argument parsing
    parser = argparse.ArgumentParser(description="Filter dataset by the most frequent sirna_id and cell_type.")
    parser.add_argument("dataset_dir", type=str, help="Path to the dataset directory containing metadata.csv")

    args = parser.parse_args()
    main(args.dataset_dir)

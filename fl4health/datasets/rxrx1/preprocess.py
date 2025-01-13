import argparse
import os
from pathlib import Path

import pandas as pd


# Function to filter data based on `sirna_id` and `cell_type`
def filter_and_save_data(data: pd.DataFrame, top_sirna_ids: list[int], cell_type: str, output_path: Path) -> None:
    """Filters data for the given cell type and saves it to a CSV file."""
    filtered_data = data[(data["sirna_id"].isin(top_sirna_ids)) & (data["cell_type"] == cell_type)]
    filtered_data.to_csv(output_path, index=False)
    print(f"Data for {cell_type} saved to {output_path}")


def main(dataset_dir: Path) -> None:
    # Paths based on the dataset directory
    metadata_file = os.path.join(dataset_dir, "metadata.csv")
    output_dir = os.path.join(dataset_dir, "clients")

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    data = pd.read_csv(metadata_file)

    # Get the top 50 `sirna_id`s by frequency
    top_sirna_ids = data["sirna_id"].value_counts().head(50).index.tolist()

    # Define cell types and corresponding output file paths
    cell_types = ["RPE", "HUVEC", "HEPG2", "U2OS"]
    output_files = [os.path.join(output_dir, f"meta_data_{i+1}.csv") for i in range(len(cell_types))]

    # Filter and save data for each client
    for cell_type, output_path in zip(cell_types, output_files):
        filter_and_save_data(data, top_sirna_ids, cell_type, Path(output_path))


if __name__ == "__main__":
    # Argument parsing
    parser = argparse.ArgumentParser(description="Filter dataset by sirna_id and cell_type.")
    parser.add_argument("dataset_dir", type=str, help="Path to the dataset directory containing metadata.csv")

    args = parser.parse_args()
    main(args.dataset_dir)

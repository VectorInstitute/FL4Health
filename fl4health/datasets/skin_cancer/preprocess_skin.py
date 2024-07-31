"""
The following code is adapted from the preprocess_skin.py script
from the medical_federated GitHub repository by Seongjun Yang et al.

Paper: https://arxiv.org/abs/2207.03075
Code: https://github.com/wns823/medical_federated.git
- medical_federated/skin_cancer_federated/preprocess_skin.py
"""

import json
import os
from typing import Any, Callable, Dict, List

import pandas as pd


def save_to_json(data: Dict[str, Any], path: str) -> None:
    """Saves a dictionary to a JSON file.

    Args:
        data: A dictionary to save.
        path: The file path to save the JSON data.
    """
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent="\t")


def process_client_data(
    dataframe: pd.DataFrame,
    client_name: str,
    data_path: str,
    image_path_func: Callable[[pd.Series], str],
    label_map_func: Callable[[pd.Series], str],
    original_columns: List[str],
    official_columns: List[str],
) -> None:
    """Processes and saves the client-specific dataset.

    Args:
        dataframe: The dataframe containing the client data.
        client_name: The name of the client.
        data_path: The base path to the dataset.
        image_path_func: A function that constructs the image path from a dataframe row.
        label_map_func: A function that maps the original label to the new label.
        original_columns: The list of original columns for the dataset.
        official_columns: The list of official columns for the dataset.
    """
    preprocessed_data: Dict[str, Any] = {
        "columns": official_columns,
        "original_columns": original_columns,
        "data": [],
    }

    for i in range(len(dataframe)):
        row = dataframe.iloc[i]
        img_path = image_path_func(row)
        label = label_map_func(row)
        origin_labels = [0] * len(original_columns)
        extended_labels = [0] * len(official_columns)
        origin_labels[original_columns.index(label)] = 1
        extended_labels[official_columns.index(label)] = 1
        preprocessed_data["data"].append(
            {
                "img_path": img_path,
                "origin_labels": origin_labels,
                "extended_labels": extended_labels,
            }
        )

    save_to_json(preprocessed_data, os.path.join(data_path, f"{client_name}.json"))


def preprocess_isic_2019(data_path: str, official_columns: List[str]) -> None:
    """Preprocesses the ISIC 2019 dataset.

    Args:
        data_path: The base path to the dataset.
        official_columns: The list of official columns for the dataset.
    """
    Isic_2019_path = os.path.join(data_path, "ISIC_2019")
    Isic_csv_path = os.path.join(Isic_2019_path, "ISIC_2019_Training_GroundTruth.csv")
    Isic_df = pd.read_csv(Isic_csv_path)

    Isic_meta = pd.read_csv(os.path.join(Isic_2019_path, "ISIC_2019_Training_Metadata.csv"))
    barcelona_list = [i for i in Isic_meta["lesion_id"].dropna() if "BCN" in i]
    barcelona_core = Isic_meta[Isic_meta["lesion_id"].isin(barcelona_list)]
    core_2019 = Isic_df[Isic_df["image"].isin(barcelona_core["image"])]
    core_2019.to_csv(os.path.join(Isic_2019_path, "ISIC_2019_core.csv"), mode="w")

    Isic_2019_data_path = os.path.join(data_path, "ISIC_2019", "ISIC_2019_Training_Input")
    Barcelona_df = pd.read_csv(os.path.join(Isic_2019_path, "ISIC_2019_core.csv"))
    Barcelona_new = Barcelona_df[["image"] + official_columns + ["UNK"]]
    preprocessed_data: Dict[str, Any] = {
        "columns": official_columns,
        "original_columns": official_columns,
        "data": [],
    }

    for i in range(len(Barcelona_new)):
        # Extract the row values, leaving off the last element ("UNK" column)
        temp = list(Barcelona_new.loc[i].values[:-1])
        img_path = os.path.join(Isic_2019_data_path, temp[0] + ".jpg")
        origin_labels = temp[1:]
        extended_labels = temp[1:]
        preprocessed_data["data"].append(
            {
                "img_path": img_path,
                "origin_labels": origin_labels,
                "extended_labels": extended_labels,
            }
        )

    save_to_json(preprocessed_data, os.path.join(data_path, "ISIC_2019", "ISIC_19_Barcelona.json"))


def ham_image_path_func(row: pd.Series) -> str:
    """Constructs the image path for the HAM10000 dataset.

    Args:
        row: A row from the dataframe.

    Returns:
        The constructed image path.
    """
    return os.path.join("fl4health", "datasets", "skin_cancer", "HAM10000", row["image_id"] + ".jpg")


def ham_label_map_func(row: pd.Series) -> str:
    """Maps the original label to the new label for the HAM10000 dataset.

    Args:
        row: A row from the dataframe.

    Returns:
        The mapped label.
    """
    Ham_labelmap = {
        "akiec": "AK",
        "bcc": "BCC",
        "bkl": "BKL",
        "df": "DF",
        "mel": "MEL",
        "nv": "NV",
        "vasc": "VASC",
    }
    return Ham_labelmap[row["dx"]]


def preprocess_ham10000(data_path: str, official_columns: List[str]) -> None:
    """Preprocesses the HAM10000 dataset.

    Args:
        data_path: The base path to the dataset.
        official_columns: The list of official columns for the dataset.
    """
    Ham_10000_path = os.path.join(data_path, "HAM10000")
    Ham_csv_path = os.path.join(Ham_10000_path, "HAM10000_metadata")
    HAM_df = pd.read_csv(Ham_csv_path)

    Rosendahl_data = HAM_df[HAM_df["dataset"] == "rosendahl"]
    Rosendahl_data.to_csv(os.path.join(Ham_10000_path, "HAM_rosendahl.csv"), mode="w")
    Vienna_data = HAM_df[HAM_df["dataset"] != "rosendahl"]
    Vienna_data.to_csv(os.path.join(Ham_10000_path, "HAM_vienna.csv"), mode="w")

    Ham_columns = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC"]

    process_client_data(
        pd.read_csv(os.path.join(Ham_10000_path, "HAM_rosendahl.csv")),
        "HAM_rosendahl",
        Ham_10000_path,
        ham_image_path_func,
        ham_label_map_func,
        Ham_columns,
        official_columns,
    )
    process_client_data(
        pd.read_csv(os.path.join(Ham_10000_path, "HAM_vienna.csv")),
        "HAM_vienna",
        Ham_10000_path,
        ham_image_path_func,
        ham_label_map_func,
        Ham_columns,
        official_columns,
    )


def pad_image_path_func(row: pd.Series) -> str:
    """Constructs the image path for the PAD-UFES-20 dataset.

    Args:
        row: A row from the dataframe.

    Returns:
        The constructed image path.
    """
    return os.path.join("fl4health", "datasets", "skin_cancer", "PAD-UFES-20", row["img_id"])


def pad_label_map_func(row: pd.Series) -> str:
    """Maps the original label to the new label for the PAD-UFES-20 dataset.

    Args:
        row: A row from the dataframe.

    Returns:
        The mapped label.
    """
    Pad_ufes_20_labelmap = {
        "ACK": "AK",
        "BCC": "BCC",
        "MEL": "MEL",
        "NEV": "NV",
        "SCC": "SCC",
        "SEK": "BKL",
    }
    return Pad_ufes_20_labelmap[row["diagnostic"]]


def preprocess_pad_ufes_20(data_path: str, official_columns: List[str]) -> None:
    """Preprocesses the PAD-UFES-20 dataset.

    Args:
        data_path: The base path to the dataset.
        official_columns: The list of official columns for the dataset.
    """
    Pad_ufes_20_path = os.path.join(data_path, "PAD-UFES-20")
    Pad_ufes_20_csv_path = os.path.join(Pad_ufes_20_path, "metadata.csv")
    Pad_ufes_20_df = pd.read_csv(Pad_ufes_20_csv_path)

    Pad_columns = ["MEL", "NV", "BCC", "AK", "BKL", "SCC"]

    process_client_data(
        Pad_ufes_20_df,
        "PAD_UFES_20",
        Pad_ufes_20_path,
        pad_image_path_func,
        pad_label_map_func,
        Pad_columns,
        official_columns,
    )


def derm7pt_image_path_func(row: pd.Series) -> str:
    """Constructs the image path for the Derm7pt dataset.

    Args:
        row: A row from the dataframe.

    Returns:
        The constructed image path.
    """
    return os.path.join("fl4health", "datasets", "skin_cancer", "Derm7pt", "images", row["derm"])


def derm7pt_label_map_func(row: pd.Series) -> str:
    """Maps the original label to the new label for the Derm7pt dataset.

    Args:
        row: A row from the dataframe.

    Returns:
        The mapped label.
    """
    Derm7pt_labelmap = {
        "basal cell carcinoma": "BCC",
        "blue nevus": "NV",
        "clark nevus": "NV",
        "combined nevus": "NV",
        "congenital nevus": "NV",
        "dermal nevus": "NV",
        "dermatofibroma": "DF",  # MISC
        "lentigo": "MISC",
        "melanoma": "MEL",
        "melanoma (0.76 to 1.5 mm)": "MEL",
        "melanoma (in situ)": "MEL",
        "melanoma (less than 0.76 mm)": "MEL",
        "melanoma (more than 1.5 mm)": "MEL",
        "melanoma metastasis": "MEL",
        "melanosis": "MISC",
        "miscellaneous": "MISC",
        "recurrent nevus": "NV",
        "reed or spitz nevus": "NV",
        "seborrheic keratosis": "BKL",
        "vascular lesion": "VASC",  # MISC
    }
    return Derm7pt_labelmap[row["diagnosis"]]


def preprocess_derm7pt(data_path: str, official_columns: List[str]) -> None:
    """Preprocesses the Derm7pt dataset.

    Args:
        data_path: The base path to the dataset.
        official_columns: The list of official columns for the dataset.
    """
    Derm7pt_path = os.path.join(data_path, "Derm7pt")
    Derm7pt_df = pd.read_csv(os.path.join(Derm7pt_path, "meta", "meta_core.csv"))

    Derm7pt_columns = ["MEL", "NV", "BCC", "BKL", "DF", "VASC"]

    process_client_data(
        Derm7pt_df,
        "Derm7pt",
        Derm7pt_path,
        derm7pt_image_path_func,
        derm7pt_label_map_func,
        Derm7pt_columns,
        official_columns,
    )


if __name__ == "__main__":
    data_path = os.path.join("fl4health", "datasets", "skin_cancer")
    official_columns = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]

    preprocess_isic_2019(data_path, official_columns)
    preprocess_ham10000(data_path, official_columns)
    preprocess_pad_ufes_20(data_path, official_columns)
    preprocess_derm7pt(data_path, official_columns)

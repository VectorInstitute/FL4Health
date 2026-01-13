"""
The following code is adapted from the ``preprocess_skin.py`` script
from the medical_federated GitHub repository by Seongjun Yang et al.

Paper: https://arxiv.org/abs/2207.03075
Code: https://github.com/wns823/medical_federated.git
- medical_federated/skin_cancer_federated/preprocess_skin.py
"""

import json
import os
from collections.abc import Callable
from typing import Any

import pandas as pd


def save_to_json(data: dict[str, Any], path: str) -> None:
    """
    Saves a dictionary to a JSON file.

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
    original_columns: list[str],
    official_columns: list[str],
) -> None:
    """
    Processes and saves the client-specific dataset.

    Args:
        dataframe: The dataframe containing the client data.
        client_name: The name of the client.
        data_path: The base path to the dataset.
        image_path_func: A function that constructs the image path from a dataframe row.
        label_map_func: A function that maps the original label to the new label.
        original_columns: The list of original columns for the dataset.
        official_columns: The list of official columns for the dataset.
    """
    preprocessed_data: dict[str, Any] = {
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


def preprocess_isic_2019(data_path: str, official_columns: list[str]) -> None:
    """
    Preprocesses the ISIC 2019 dataset.

    Args:
        data_path (str): The base path to the dataset.
        official_columns (list[str]): The list of official columns for the dataset.
    """
    isic_2019_path = os.path.join(data_path, "ISIC_2019")
    isic_csv_path = os.path.join(isic_2019_path, "ISIC_2019_Training_GroundTruth.csv")
    isic_df = pd.read_csv(isic_csv_path)

    isic_meta = pd.read_csv(os.path.join(isic_2019_path, "ISIC_2019_Training_Metadata.csv"))
    barcelona_list = [i for i in isic_meta["lesion_id"].dropna() if "BCN" in i]
    barcelona_core = isic_meta[isic_meta["lesion_id"].isin(barcelona_list)]
    core_2019 = isic_df[isic_df["image"].isin(barcelona_core["image"])]
    core_2019.to_csv(os.path.join(isic_2019_path, "ISIC_2019_core.csv"), mode="w")

    isic_2019_data_path = os.path.join(data_path, "ISIC_2019", "ISIC_2019_Training_Input")
    barcelona_df = pd.read_csv(os.path.join(isic_2019_path, "ISIC_2019_core.csv"))
    barcelona_new = barcelona_df[["image"] + official_columns + ["UNK"]]
    preprocessed_data: dict[str, Any] = {
        "columns": official_columns,
        "original_columns": official_columns,
        "data": [],
    }

    for i in range(len(barcelona_new)):
        # Extract the row values, leaving off the last element ("UNK" column)
        temp = list(barcelona_new.loc[i].values[:-1])
        img_path = os.path.join(isic_2019_data_path, temp[0] + ".jpg")
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
    """
    Constructs the image path for the HAM10000 dataset.

    Args:
        row (pd.Series): A row from the dataframe.

    Returns:
        (str): The constructed image path.
    """
    return os.path.join("fl4health", "datasets", "skin_cancer", "HAM10000", row["image_id"] + ".jpg")


def ham_label_map_func(row: pd.Series) -> str:
    """
    Maps the original label to the new label for the HAM10000 dataset.

    Args:
        row (pd.Series): A row from the dataframe.

    Returns:
        (str): The mapped label.
    """
    ham_labelmap = {
        "akiec": "AK",
        "bcc": "BCC",
        "bkl": "BKL",
        "df": "DF",
        "mel": "MEL",
        "nv": "NV",
        "vasc": "VASC",
    }
    return ham_labelmap[row["dx"]]


def preprocess_ham10000(data_path: str, official_columns: list[str]) -> None:
    """
    Preprocesses the HAM10000 dataset.

    Args:
        data_path (str): The base path to the dataset.
        official_columns (list[str]): The list of official columns for the dataset.
    """
    ham_10000_path = os.path.join(data_path, "HAM10000")
    ham_csv_path = os.path.join(ham_10000_path, "HAM10000_metadata")
    ham_df = pd.read_csv(ham_csv_path)

    rosendahl_data = ham_df[ham_df["dataset"] == "rosendahl"]
    rosendahl_data.to_csv(os.path.join(ham_10000_path, "HAM_rosendahl.csv"), mode="w")
    vienna_data = ham_df[ham_df["dataset"] != "rosendahl"]
    vienna_data.to_csv(os.path.join(ham_10000_path, "HAM_vienna.csv"), mode="w")

    ham_columns = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC"]

    process_client_data(
        pd.read_csv(os.path.join(ham_10000_path, "HAM_rosendahl.csv")),
        "HAM_rosendahl",
        ham_10000_path,
        ham_image_path_func,
        ham_label_map_func,
        ham_columns,
        official_columns,
    )
    process_client_data(
        pd.read_csv(os.path.join(ham_10000_path, "HAM_vienna.csv")),
        "HAM_vienna",
        ham_10000_path,
        ham_image_path_func,
        ham_label_map_func,
        ham_columns,
        official_columns,
    )


def pad_image_path_func(row: pd.Series) -> str:
    """
    Constructs the image path for the PAD-UFES-20 dataset.

    Args:
        row (pd.Series): A row from the dataframe.

    Returns:
        (str): The constructed image path.
    """
    return os.path.join("fl4health", "datasets", "skin_cancer", "PAD-UFES-20", row["img_id"])


def pad_label_map_func(row: pd.Series) -> str:
    """
    Maps the original label to the new label for the PAD-UFES-20 dataset.

    Args:
        row (pd.Series): A row from the dataframe.

    Returns:
        (str): The mapped label.
    """
    pad_ufes_20_labelmap = {
        "ACK": "AK",
        "BCC": "BCC",
        "MEL": "MEL",
        "NEV": "NV",
        "SCC": "SCC",
        "SEK": "BKL",
    }
    return pad_ufes_20_labelmap[row["diagnostic"]]


def preprocess_pad_ufes_20(data_path: str, official_columns: list[str]) -> None:
    """
    Preprocesses the PAD-UFES-20 dataset.

    Args:
        data_path (str): The base path to the dataset.
        official_columns (list[str]): The list of official columns for the dataset.
    """
    pad_ufes_20_path = os.path.join(data_path, "PAD-UFES-20")
    pad_ufes_20_csv_path = os.path.join(pad_ufes_20_path, "metadata.csv")
    pad_ufes_20_df = pd.read_csv(pad_ufes_20_csv_path)

    pad_columns = ["MEL", "NV", "BCC", "AK", "BKL", "SCC"]

    process_client_data(
        pad_ufes_20_df,
        "PAD_UFES_20",
        pad_ufes_20_path,
        pad_image_path_func,
        pad_label_map_func,
        pad_columns,
        official_columns,
    )


def derm7pt_image_path_func(row: pd.Series) -> str:
    """
    Constructs the image path for the Derm7pt dataset.

    Args:
        row (pd.Series): A row from the dataframe.

    Returns:
        (str):  The constructed image path.
    """
    return os.path.join("fl4health", "datasets", "skin_cancer", "Derm7pt", "images", row["derm"])


def derm7pt_label_map_func(row: pd.Series) -> str:
    """
    Maps the original label to the new label for the Derm7pt dataset.

    Args:
        row (pd.Series): A row from the dataframe.

    Returns:
        (str):  The mapped label.
    """
    derm7pt_labelmap = {
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
    return derm7pt_labelmap[row["diagnosis"]]


def preprocess_derm7pt(data_path: str, official_columns: list[str]) -> None:
    """
    Preprocesses the Derm7pt dataset.

    Args:
        data_path (str): The base path to the dataset.
        official_columns (list[str]): The list of official columns for the dataset.
    """
    derm7pt_path = os.path.join(data_path, "Derm7pt")
    derm7pt_df = pd.read_csv(os.path.join(derm7pt_path, "meta", "meta_core.csv"))

    derm7pt_columns = ["MEL", "NV", "BCC", "BKL", "DF", "VASC"]

    process_client_data(
        derm7pt_df,
        "Derm7pt",
        derm7pt_path,
        derm7pt_image_path_func,
        derm7pt_label_map_func,
        derm7pt_columns,
        official_columns,
    )


if __name__ == "__main__":
    data_path = os.path.join("fl4health", "datasets", "skin_cancer")
    official_columns = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]

    preprocess_isic_2019(data_path, official_columns)
    preprocess_ham10000(data_path, official_columns)
    preprocess_pad_ufes_20(data_path, official_columns)
    preprocess_derm7pt(data_path, official_columns)

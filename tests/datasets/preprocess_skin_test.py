import json
import os
from pathlib import Path

import pandas as pd

from fl4health.datasets.skin_cancer.preprocess_skin import (
    derm7pt_image_path_func,
    derm7pt_label_map_func,
    ham_image_path_func,
    ham_label_map_func,
    pad_image_path_func,
    pad_label_map_func,
    save_to_json,
)


TEST_PD_SERIES1 = pd.Series({"one": 1.0, "two": "two", "three": [1.0, 2.0], "image_id": "test_id", "dx": "bkl"})
TEST_PD_SERIES2 = pd.Series(
    {"one": 1.0, "two": "two", "three": [1.0, 2.0], "img_id": "test_id.jpg", "diagnostic": "ACK"}
)
TEST_PD_SERIES3 = pd.Series(
    {
        "one": 1.0,
        "two": "two",
        "three": [1.0, 2.0],
        "derm": "test_image.jpg",
        "diagnosis": "melanoma (more than 1.5 mm)",
    }
)

TEST_PD_SERIES4 = pd.Series({"one": 1.0, "two": "two", "three": [1.0, 2.0], "image_id": "test_id", "dx": "bcc"})
TEST_PD_SERIES5 = pd.Series(
    {"one": 1.0, "two": "two", "three": [1.0, 2.0], "img_id": "test_id.jpg", "diagnostic": "SEK"}
)
TEST_PD_SERIES6 = pd.Series(
    {"one": 1.0, "two": "two", "three": [1.0, 2.0], "derm": "test_image.jpg", "diagnosis": "dermal nevus"}
)


def test_save_to_json(tmp_path: Path) -> None:
    save_path = os.path.join(tmp_path, "test_json.json")
    dict_to_save = {"one": 1.0, "two": "two", "three": [1.0, 2.0]}
    save_to_json(dict_to_save, save_path)

    with open(save_path, "r") as f:
        loaded_dict = json.load(f)

    for key, val in dict_to_save.items():
        assert val == loaded_dict[key]


def test_path_functions() -> None:
    path = ham_image_path_func(TEST_PD_SERIES1)
    assert str(path) == "fl4health/datasets/skin_cancer/HAM10000/test_id.jpg"

    path = pad_image_path_func(TEST_PD_SERIES2)
    assert str(path) == "fl4health/datasets/skin_cancer/PAD-UFES-20/test_id.jpg"

    path = derm7pt_image_path_func(TEST_PD_SERIES3)
    assert str(path) == "fl4health/datasets/skin_cancer/Derm7pt/images/test_image.jpg"


def test_label_map_functions() -> None:
    label = ham_label_map_func(TEST_PD_SERIES1)
    assert str(label) == "BKL"

    label = ham_label_map_func(TEST_PD_SERIES4)
    assert str(label) == "BCC"

    label = pad_label_map_func(TEST_PD_SERIES2)
    assert str(label) == "AK"

    label = pad_label_map_func(TEST_PD_SERIES5)
    assert str(label) == "BKL"

    label = derm7pt_label_map_func(TEST_PD_SERIES3)
    assert str(label) == "MEL"

    label = derm7pt_label_map_func(TEST_PD_SERIES6)
    assert str(label) == "NV"

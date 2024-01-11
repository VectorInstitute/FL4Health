
import json
from pathlib import Path
from typing import List, Any, Optional, Tuple, Callable
import torch
from batchgenerators.dataloading.data_loader import DataLoader
from collections import OrderedDict
import numpy as np
from numpy import typing as npt
from monai.transforms.compose import Compose
from monai.transforms.utility.array import EnsureType

from research.picai.dataset import SimpleITKDataset


# TODO: Pin type annotations to Union of expected types once decide on accepted inputs.
def default_collate(batch: List[Any]) -> Any:
    """
    Function that groups set of data into a batch based on type of data in batch and
    structure that its stored.

    Args:
        batch (List[Any]): A list of data in which every entry is associated with a sample
        (or corresponding label) in the batch.

    Returns:
        Any: Batch of data.

    """
    if isinstance(batch[0], np.ndarray):
        return np.vstack(batch)
    elif isinstance(batch[0], (int, np.int64)):
        return np.array(batch).astype(np.int32)
    elif isinstance(batch[0], (float, np.float32)):
        return np.array(batch).astype(np.float32)
    elif isinstance(batch[0], (np.float64,)):
        return np.array(batch).astype(np.float64)
    elif isinstance(batch[0], (dict, OrderedDict)):
        return {key: default_collate([d[key] for d in batch]) for key in batch[0]}
    elif isinstance(batch[0], (tuple, list)):
        transposed = zip(*batch)
        return [default_collate(samples) for samples in transposed]
    elif isinstance(batch[0], str):
        return batch
    elif isinstance(batch[0], torch.Tensor):
        return torch.vstack(batch)
    else:
        raise TypeError('unknown type for batch:', type(batch))


# TODO: Pin type annotations to Union of expected types once decide on accepted inputs.
class DataLoaderFromDataset(DataLoader):
    def __init__(self, data: SimpleITKDataset, batch_size: int, num_threads: int, seed_for_shuffle: int = 1, collate_fn: Callable = default_collate,
                 return_incomplete: bool = False, shuffle: bool = True, infinite: bool = False) -> None:
        """
        Extends batchgenerators DataLoader class by implementing generate_train_batch method to yield a valid
        PyTorch DataLoader in which augemntations can be applied following its creation.

        Args:
            data (SimpleITKDataset): The dataset the DataLoader is being created for.
            batch_size (int): Size of the batch for the DataLoader.
            num_threads (int): Number of threads used to load data.
            seed_for_shuffle (int): The random used for shuffling data.
            collate_fn (Callable): Function that yields groups a set of data into a batch.
        """

        super(DataLoaderFromDataset, self).__init__(data, batch_size, num_threads, seed_for_shuffle,
                                                    return_incomplete=return_incomplete, shuffle=shuffle,
                                                    infinite=infinite)
        self.collate_fn = collate_fn
        self.indices = np.arange(len(data))

    def generate_train_batch(self) -> Any:
        """
        Generates a batch by sampling indices equal to the batch size, indexing into the data and labels, inserting them into 
        into a sample wise dictionary and grouping in a list. 

        Returns:
            Any: A batch of data that has been created via the collate fn. 
        """
        # randomly select N samples (N = batch size)
        indices = self.get_indices()

        # create dictionary per sample
        batch = [{'data': self._data[i][0].numpy(),
                  'seg': self._data[i][1].numpy()} for i in indices]

        return self.collate_fn(batch)

def z_score_norm(image: "npt.NDArray[Any]", percentile: Optional[float] = None) -> "npt.NDArray[Any]":
    """
    Function that performs instance wise Z-score normalization (mean=0; stdev=1), where intensities
    below or above the given percentile are discarded.

    Args:
        image (npt.Ndarray[Any]): N-dimensional image to be normalized and optionally clipped. 
        percentile (Optional[float]): Percentile used to set threshold to clip activations.
            If None, no clipping occurs. If a percentile is specified, must be 0 =< 50

    Returns:
       npt.NDArray[Any]: Z-Score Normalized vesrion of input that is clipped if a percentile is specified. 
    """
    image = image.astype(np.float32)

    if percentile is not None:
        assert (percentile >= 0 and percentile <= 50)
        # clip distribution of intensity values
        lower_bnd = np.percentile(image, 100-percentile)
        upper_bnd = np.percentile(image, percentile)
        image = np.clip(image, lower_bnd, upper_bnd)

    # perform z-score normalization
    mean = np.mean(image)
    std = np.std(image)
    if std > 0:
        return (image - mean) / std
    else:
        return image * 0.

def get_dataloaders(overview_dir: str, batch_size: int, num_threads: int, fold_id: int) -> Tuple[DataLoader, DataLoader, "npt.NDArray[np.float32]"]:
    """
    Function that initializes and returns the train and validation DataLoader along with proportion of samples
    with each label.

    Args:
        overview_dir (str): The path to the directory that houses the datasheets for the train and validation data.
        batch_size (str): The number of samples per batch yielded by the DataLoader.
        num_threads (int): The number of threads used by the DataLoader. 
        fold_id (int): The cross validation fold to use to create the DataLoader.

    Returns:
       Tuple[DataLoader, DataLoader, np.array]: The Training DataLoader, Validation Loader and Numpy Array
       with the proportion of samples in each class.
        
    """
    # load datasheets
    with open(Path(overview_dir) / f'PI-CAI_train-fold-{fold_id}.json') as fp:
        train_json = json.load(fp)
    with open(Path(overview_dir) / f'PI-CAI_val-fold-{fold_id}.json') as fp:
        valid_json = json.load(fp)

    # load paths to images and labels
    train_data = [np.array(train_json['image_paths']), np.array(train_json['label_paths'])]
    valid_data = [np.array(valid_json['image_paths']), np.array(valid_json['label_paths'])]

    # use case-level class balance to deduce required train-time class weights
    class_ratio_t = [int(np.sum(train_json['case_label'])), int(len(train_data[0])-np.sum(train_json['case_label']))]
    class_ratio_v = [int(np.sum(valid_json['case_label'])), int(len(valid_data[0])-np.sum(valid_json['case_label']))]
    class_weights = (class_ratio_t / np.sum(class_ratio_t))

    # log dataset definition
    print('Dataset Definition:', "-"*80)
    print(f'Fold Number: {fold_id}')
    print('Data Classes:', list(np.unique(train_json['case_label'])))
    print(f'Train-Time Class Weights: {class_weights}')
    print(f'Training Samples [-:{class_ratio_t[1]};+:{class_ratio_t[0]}]: {len(train_data[1])}')
    print(f'Validation Samples [-:{class_ratio_v[1]};+:{class_ratio_v[0]}]: {len(valid_data[1])}')

    # actual dataloaders used at train-time
    pretx = [EnsureType()]
    train_ds = SimpleITKDataset(image_files=str(train_data[0]), seg_files=str(train_data[1]),
                                transform=Compose(pretx),  seg_transform=Compose(pretx))
    valid_ds = SimpleITKDataset(image_files=str(valid_data[0]), seg_files=str(valid_data[1]),
                                transform=Compose(pretx),  seg_transform=Compose(pretx))
    train_loader = DataLoaderFromDataset(train_ds, 
        batch_size=batch_size, num_threads=num_threads, infinite=True, shuffle=True)
    val_loader = DataLoaderFromDataset(valid_ds, 
        batch_size=batch_size, num_threads=num_threads, infinite=False, shuffle=False)

    return train_loader, val_loader, class_weights.astype(np.float32)

from functools import partial
from pathlib import Path
from typing import Any, Callable, Optional

import torch
from torch.utils.data.dataloader import DataLoader

from fl4health.model_bases.pca import PCAModule
from fl4health.utils.dataset import BaseDataset
from fl4health.utils.sampler import LabelBasedSampler


class PCAPreprocessor:
    def __init__(self, checkpointing_path: Path) -> None:
        """
        Class that leverages pre-computed principal components of
        a dataset to perform data-preprocessing.

        Args:
            checkpointing_path (Path): Path to saved principal components.
        """
        self.checkpointing_path = checkpointing_path
        self.pca_module: PCAModule = self.load_pca_module()

    def load_pca_module(self) -> PCAModule:
        pca_module = torch.load(self.checkpointing_path)
        pca_module.eval()
        return pca_module

    def reduce_dimension(
        self,
        new_dimension: int,
        batch_size: int,
        shuffle: bool,
        sampler: Optional[LabelBasedSampler],
        get_dataset: Callable[..., BaseDataset],
        *args: Any,
    ) -> DataLoader:
        """
        Perform dimensionality reduction on a dataset by projecting the data
        onto a set of pre-computed principal components.

        Args:
            new_dimension (int): New data dimension after dimensionality reduction. Equals
            the number of principal components onto which projection is performed.
            batch_size (int): Batch size.
            shuffle (bool): Whether the dataloader should be shuffled.
            sampler (Optional[LabelBasedSampler]): For performing subsampling on the dataset if needed.
            get_dataset (Callable[..., BaseDataset]): Function that returns an instance of BaseDataset.

        Returns:
            DataLoader: a dataloader consisting of data with reduced dimension.
        """
        projection = partial(self.pca_module.project_lower_dim, k=new_dimension)
        dataset = get_dataset(*args)
        if sampler is not None:
            dataset = sampler.subsample(dataset)
        dataset.update_transform(projection)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

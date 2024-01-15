from functools import partial
from pathlib import Path

import torch

from fl4health.model_bases.pca import PcaModule
from fl4health.utils.dataset import BaseDataset


class PcaPreprocessor:
    def __init__(self, checkpointing_path: Path) -> None:
        """
        Class that leverages pre-computed principal components of
        a dataset to perform data-preprocessing.

        Args:
            checkpointing_path (Path): Path to saved principal components.
        """
        self.checkpointing_path = checkpointing_path
        self.pca_module: PcaModule = self.load_pca_module()

    def load_pca_module(self) -> PcaModule:
        pca_module = torch.load(self.checkpointing_path)
        pca_module.eval()
        return pca_module

    def reduce_dimension(
        self,
        new_dimension: int,
        dataset: BaseDataset,
    ) -> BaseDataset:
        """
        Perform dimensionality reduction on a dataset by projecting the data
        onto a set of pre-computed principal components.

        (Note that PyTorch dataloaders perform lazy application of transforms.
        So in reality, dimensionality reduction is applied in real-time as the user
        iterates through the dataloader created from the dataset returned here.)

        Args:
            new_dimension (int): New data dimension after dimensionality reduction. Equals
            the number of principal components onto which projection is performed.
            dataset (BaseDataset): Dataset containing data whose dimension is to be reduced.
        Returns:
            BaseDataset: Dataset consisting of data with reduced dimension.
        """
        projection = partial(self.pca_module.project_lower_dim, k=new_dimension)
        dataset.update_transform(projection)
        return dataset

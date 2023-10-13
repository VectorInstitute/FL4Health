from logging import INFO
from pathlib import Path
from typing import Dict, Sequence

import pandas as pd
import torch
from flwr.common.logger import log
from flwr.common.typing import Config, NDArray, Scalar

from fl4health.clients.basic_client import BasicClient
from fl4health.feature_alignment.constants import (
    BINARY,
    FEATURE_INFO,
    FORMAT_SPECIFIED,
    INPUT_DIMENSION,
    ORDINAL,
    OUTPUT_DIMENSION,
)
from fl4health.feature_alignment.tab_features_info_encoder import TabularFeaturesInfoEncoder
from fl4health.feature_alignment.tab_features_preprocessor import TabularFeaturesPreprocessor
from fl4health.utils.metrics import Metric


class TabularDataClient(BasicClient):
    def __init__(
        self, data_path: Path, metrics: Sequence[Metric], device: torch.device, id_column: str, target_column: str
    ) -> None:
        super().__init__(data_path, metrics, device)
        self.tabular_features_info_encoder: TabularFeaturesInfoEncoder
        self.tabular_features_preprocessor: TabularFeaturesPreprocessor
        self.df: pd.DataFrame
        self.input_dimension: int
        self.output_dimension: int
        self.id_column = id_column
        self.target_column = target_column
        # The aligned data and targets, which are used to construct dataloaders.
        self.aligned_features: NDArray
        self.aligned_targets: NDArray

    def setup_client(self, config: Config) -> None:
        """
        Initialize the client by encoding the information of its tabular data
        and initializing the corresponding TabularFeaturesPreprocessor.

        config[FORMAT_SPECIFIED] indicates whether the server has obtained
        the source of information to perform feature alignment.
        If it is True, it means the server has obtained such information
        (either a priori or by polling a client).
        So the client will encode that information and use it instead
        to perform feature preprocessing.
        """
        assert FORMAT_SPECIFIED in config.keys()
        format_specified = self.narrow_config_type(config, FORMAT_SPECIFIED, bool)

        # Encode the information of the client's local tabular data.
        self.df = self.get_data_frame(config)
        self.tabular_features_info_encoder = TabularFeaturesInfoEncoder.encoder_from_dataframe(
            self.df, self.id_column, self.target_column
        )

        if format_specified:
            # Since the server has obtained its source of information,
            # the client will encode that instead.
            self.tabular_features_info_encoder = TabularFeaturesInfoEncoder.from_json(
                self.narrow_config_type(config, FEATURE_INFO, str)
            )
            self.tabular_features_preprocessor = TabularFeaturesPreprocessor(self.tabular_features_info_encoder)

            # preprocess features.
            self.aligned_features, self.aligned_targets = self.tabular_features_preprocessor.preprocess_features(
                self.df
            )

            # Obtain the input and output dimensions to be sent to
            # the server for global model initialization.
            self.input_dimension = self.aligned_features.shape[1]
            target_type = self.tabular_features_info_encoder.get_target_type()
            if target_type == ORDINAL or target_type == BINARY:
                self.output_dimension = len(self.tabular_features_info_encoder.get_target_categories())
            else:
                self.output_dimension = 1
            log(INFO, f"input dimension: {self.input_dimension}, output_dimension: {self.output_dimension}")

            super().setup_client(config)

    def get_data_frame(self, config: Config) -> pd.DataFrame:
        """
        User defined method that returns a pandas dataframe.
        """
        raise NotImplementedError

    def get_properties(self, config: Config) -> Dict[str, Scalar]:
        """
        Return properties of client to be sent to the server.
        Depending on whether the server has communicated the information
        to be used for feature alignment, the client will send the input/output
        dimensions so the server can use them to initialize the global model.

        First initializes the client because this is called prior to the first
        federated learning round.
        """
        self.setup_client(config)
        format_specified = self.narrow_config_type(config, FORMAT_SPECIFIED, bool)
        if not format_specified:
            return {
                FEATURE_INFO: self.tabular_features_info_encoder.to_json(),
            }
        else:
            return {
                FEATURE_INFO: self.tabular_features_info_encoder.to_json(),
                INPUT_DIMENSION: self.input_dimension,
                OUTPUT_DIMENSION: self.output_dimension,
            }

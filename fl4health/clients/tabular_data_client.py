from collections.abc import Sequence
from logging import INFO
from pathlib import Path

import pandas as pd
import torch
from flwr.common.logger import log
from flwr.common.typing import Config, NDArray, Scalar
from sklearn.pipeline import Pipeline

from fl4health.checkpointing.client_module import ClientCheckpointAndStateModule
from fl4health.clients.basic_client import BasicClient
from fl4health.feature_alignment.constants import FEATURE_INFO, INPUT_DIMENSION, OUTPUT_DIMENSION, SOURCE_SPECIFIED
from fl4health.feature_alignment.tab_features_info_encoder import TabularFeaturesInfoEncoder
from fl4health.feature_alignment.tab_features_preprocessor import TabularFeaturesPreprocessor
from fl4health.metrics.base_metrics import Metric
from fl4health.reporting.base_reporter import BaseReporter
from fl4health.utils.config import narrow_dict_type
from fl4health.utils.losses import LossMeterType


class TabularDataClient(BasicClient):
    def __init__(
        self,
        data_path: Path,
        metrics: Sequence[Metric],
        device: torch.device,
        id_column: str,
        targets: str | list[str],
        loss_meter_type: LossMeterType = LossMeterType.AVERAGE,
        checkpoint_and_state_module: ClientCheckpointAndStateModule | None = None,
        reporters: Sequence[BaseReporter] | None = None,
        progress_bar: bool = False,
        client_name: str | None = None,
    ) -> None:
        """
        Client to facilitate federated feature space alignment, specifically for tabular data, and then perform
        federated training.

        Args:
            data_path (Path): path to the data to be used to load the data for client-side training.
            metrics (Sequence[Metric]): Metrics to be computed based on the labels and predictions of the client model.
            device (torch.device): Device indicator for where to send the model, batches, labels etc. Often "cpu" or
                "cuda".
            id_column (str): ID column. This is required for tabular encoding. It should be unique per row, but need
                not necessarily be a meaningful identifier (i.e. could be row number)
            targets (str | list[str]): The target column or columns name. This allows for multiple targets to
                be specified if desired.
            loss_meter_type (LossMeterType, optional): Type of meter used to track and compute the losses over
                each batch. Defaults to ``LossMeterType.AVERAGE``.
            checkpoint_and_state_module (ClientCheckpointAndStateModule | None, optional): A module meant to handle
                both checkpointing and state saving. The module, and its underlying model and state checkpointing
                components will determine when and how to do checkpointing during client-side training.
                No checkpointing (state or model) is done if not provided. Defaults to None.
            reporters (Sequence[BaseReporter] | None, optional): A sequence of FL4Health reporters which the client
                should send data to. Defaults to None.
            progress_bar (bool, optional): Whether or not to display a progress bar during client training and
                validation. Uses ``tqdm``. Defaults to False
            client_name (str | None, optional): An optional client name that uniquely identifies a client.
                If not passed, a hash is randomly generated. Client state will use this as part of its state file
                name. Defaults to None.
        """
        super().__init__(
            data_path=data_path,
            metrics=metrics,
            device=device,
            loss_meter_type=loss_meter_type,
            checkpoint_and_state_module=checkpoint_and_state_module,
            reporters=reporters,
            progress_bar=progress_bar,
            client_name=client_name,
        )
        self.tabular_features_info_encoder: TabularFeaturesInfoEncoder
        self.tabular_features_preprocessor: TabularFeaturesPreprocessor
        self.df: pd.DataFrame
        self.input_dimension: int
        self.output_dimension: int
        self.id_column = id_column
        self.targets = targets
        # The aligned data and targets, which are used to construct dataloaders.
        self.aligned_features: NDArray
        self.aligned_targets: NDArray
        self.feature_specific_pipelines: dict[str, Pipeline] = {}

    def setup_client(self, config: Config) -> None:
        """
        Initialize the client by encoding the information of its tabular data and initializing the corresponding
        ``TabularFeaturesPreprocessor``.

        ``config[SOURCE_SPECIFIED]`` indicates whether the server has obtained the source of information to perform
        feature alignment. If it is True, it means the server has obtained such information (either a priori or by
        polling a client). So the client will encode that information and use it instead to perform feature
        preprocessing.

        Args:
            config (Config): Configuration sent by the server for customization of the function
        """
        source_specified = narrow_dict_type(config, SOURCE_SPECIFIED, bool)
        self.df = self.get_data_frame(config)

        if source_specified:
            # Since the server has obtained its source of information,
            # the client will encode that instead.
            self.tabular_features_info_encoder = TabularFeaturesInfoEncoder.from_json(
                narrow_dict_type(config, FEATURE_INFO, str)
            )
            self.tabular_features_preprocessor = TabularFeaturesPreprocessor(self.tabular_features_info_encoder)

            # Set feature specific pipelines if the user has defined them.
            self.set_feature_specific_pipelines()

            # preprocess features.
            self.aligned_features, self.aligned_targets = self.tabular_features_preprocessor.preprocess_features(
                self.df
            )

            # Obtain the input and output dimensions to be sent to the server for global model initialization. Assuming
            # that the first dimension is the number of rows.
            self.input_dimension = self.aligned_features.shape[1]
            self.output_dimension = self.tabular_features_info_encoder.get_target_dimension()
            log(INFO, f"input dimension: {self.input_dimension}, target dimension: {self.output_dimension}")

            super().setup_client(config)

            # freeing the memory of aligned features/targets and data.
            del self.aligned_features
            del self.aligned_targets
            del self.df
        else:
            # Encode the information of the client's local tabular data. This is expected to happen only once
            # if the client is requested to provide alignment information.
            self.tabular_features_info_encoder = TabularFeaturesInfoEncoder.encoder_from_dataframe(
                self.df, self.id_column, self.targets
            )

    def get_data_frame(self, config: Config) -> pd.DataFrame:
        """
        User defined method that returns a pandas dataframe.

        Args:
            config (Config): Configuration sent by the server for customization of the function

        """
        raise NotImplementedError

    def get_properties(self, config: Config) -> dict[str, Scalar]:
        """
        Return properties of client to be sent to the server. Depending on whether the server has communicated the
        information to be used for feature alignment, the client will send the input/output dimensions so the server
        can use them to initialize the global model.

        First initializes the client because this is called prior to the first federated learning round.

        Args:
            config (Config): Configuration sent by the server for customization of the function

        Returns:
            (dict[str, Scalar]): Properties to be returned to the server, providing information about the client.
        """
        if not self.initialized:
            self.setup_client(config)
        source_specified = narrow_dict_type(config, SOURCE_SPECIFIED, bool)
        if not source_specified:
            return {
                FEATURE_INFO: self.tabular_features_info_encoder.to_json(),
            }
        return {
            INPUT_DIMENSION: self.input_dimension,
            OUTPUT_DIMENSION: self.output_dimension,
        }

    def preset_specific_pipeline(self, feature_name: str, pipeline: Pipeline) -> None:
        """
        The user may use this method to specify a specific pipeline to be applied to a particular feature. This
        function stores the provided pipeline associated with the provided ``feature_name``.

        Args:
            feature_name (str): Name of the feature as a column in the dataframe
            pipeline (Pipeline): Pipeline of transformations to be applied to the target feature
        """
        self.feature_specific_pipelines[feature_name] = pipeline

    def set_feature_specific_pipelines(self) -> None:
        """Given the feature specific pipelines, at them to the tabular feature preprocessor."""
        assert self.tabular_features_preprocessor is not None
        for feature_name, pipeline in self.feature_specific_pipelines.items():
            self.tabular_features_preprocessor.set_feature_pipeline(feature_name, pipeline)

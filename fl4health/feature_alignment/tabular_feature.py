from __future__ import annotations

import json

from flwr.common.typing import Scalar

from fl4health.feature_alignment.tabular_type import TabularType


MetaData = dict[str, int] | list[Scalar]


class TabularFeature:
    def __init__(
        self,
        feature_name: str,
        feature_type: TabularType,
        fill_value: Scalar | None,
        metadata: MetaData | None = None,
    ) -> None:
        """
        Information that represents a tabular feature.

        Args:
            feature_name (str): name of the feature.
            feature_type (TabularType): data type of the feature.
            fill_value (Scalar | None): the default fill value for this feature when it is missing in a dataframe.
            metadata (MetaData, optional): metadata associated with this feature.
                For example, if the feature is categorical, then metadata would be all the categories.
                Defaults to None.
        """
        self.feature_name = feature_name
        self.feature_type = feature_type
        # Each TabularType has its own default fill value, which is used
        # when the feature does not have its default fill value specified.
        if fill_value is None:
            self.fill_value = TabularType.get_default_fill_value(self.feature_type)
        else:
            self.fill_value = fill_value
        if metadata:
            self.metadata = metadata
        else:
            self.metadata = []

    def get_feature_name(self) -> str:
        return self.feature_name

    def get_feature_type(self) -> TabularType:
        return self.feature_type

    def get_fill_value(self) -> Scalar:
        return self.fill_value

    def get_metadata(self) -> MetaData:
        return self.metadata

    def get_metadata_dimension(self) -> int:
        if self.feature_type in {TabularType.BINARY, TabularType.ORDINAL}:
            return len(self.metadata)
        if self.feature_type == TabularType.NUMERIC:
            return 1
        raise ValueError("Metadata dimension is not supported when self.feature_type is TabularType.STRING.")

    def to_json(self) -> str:
        """
        Converge the information in this class to json format for serialization.

        Returns:
            str: Json with all of the pieces of information in this class
        """
        return json.dumps(
            {
                "feature_name": json.dumps(self.get_feature_name()),
                "feature_type": json.dumps(self.get_feature_type()),
                "fill_value": json.dumps(self.get_fill_value()),
                "metadata": json.dumps(self.get_metadata()),
            }
        )

    @staticmethod
    def from_json(json_str: str) -> TabularFeature:
        """
        Provided a JSON string, this function reconstructs the ``TabularFeature`` class to which it corresponds.

        Args:
            json_str (str): json string with all of the information necessary to construct the ``TabularFeature``
                object

        Returns:
            TabularFeature: Reconstructed ``TabularFeature`` object from the provided JSON
        """
        attributes = json.loads(json_str)
        return TabularFeature(
            json.loads(attributes["feature_name"]),
            TabularType(json.loads(attributes["feature_type"])),
            json.loads(attributes["fill_value"]),
            json.loads(attributes["metadata"]),
        )

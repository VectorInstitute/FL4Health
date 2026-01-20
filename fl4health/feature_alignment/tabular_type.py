from __future__ import annotations

from enum import Enum

from flwr.common.typing import Scalar


class TabularType(str, Enum):
    NUMERIC = "numeric"
    BINARY = "binary"
    ORDINAL = "ordinal"
    STRING = "string"

    @staticmethod
    def get_default_fill_value(tabular_type: TabularType | str) -> Scalar:
        """
        Provided the tabular feature type as either a string or enum, this function returns the default value for
        imputation to be used.

        Args:
            tabular_type (TabularType | str): Type of tabular feature to be imputed.

        Raises:
            ValueError: If the tabular type is unknown this will be thrown.

        Returns:
            (Scalar): Default imputation value for the specified ``TabularType``
        """
        if tabular_type is TabularType.NUMERIC:
            return 0.0
        if tabular_type is TabularType.BINARY:
            return 0
        if tabular_type is TabularType.STRING:
            return "N/A"
        if tabular_type is TabularType.ORDINAL:
            return "UNKNOWN"
        raise ValueError("Invalid Tabular Data Type.")

from enum import Enum
from typing import Union

from flwr.common.typing import Scalar


class TabularType(str, Enum):
    NUMERIC = "numeric"
    BINARY = "binary"
    ORDINAL = "ordinal"
    STRING = "string"

    @staticmethod
    def get_default_fill_value(tabular_type: Union["TabularType", str]) -> Scalar:
        if tabular_type is TabularType.NUMERIC:
            return 0.0
        elif tabular_type is TabularType.BINARY:
            return 0
        elif tabular_type is TabularType.STRING:
            return "N/A"
        elif tabular_type is TabularType.ORDINAL:
            return "UNKNOWN"
        else:
            raise ValueError("Invalid Tabular Data Type.")

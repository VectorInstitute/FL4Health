from typing import Dict, Union

from flwr.common.typing import Scalar
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer, TfidfTransformer, TfidfVectorizer

TextFeatureTransformer = Union[CountVectorizer, TfidfTransformer, TfidfVectorizer, HashingVectorizer]

# constants related to tabular data
NUMERIC = "numeric"
BINARY = "binary"
STRING = "string"
ORDINAL = "ordinal"
UNKNOWN = "UNKNOWN"

FEATURE_TYPES = [
    NUMERIC,
    BINARY,
    STRING,
    ORDINAL,
]

DEFAULT_FILL_VALUES: Dict[str, Scalar] = {BINARY: 0, NUMERIC: 0.0, ORDINAL: UNKNOWN, STRING: "N/A"}

# constants used in config for communication between
# the server and clients.
FORMAT_SPECIFIED = "format_specified"
FEATURE_INFO = "feature_info"
INPUT_DIMENSION = "input_dimension"
OUTPUT_DIMENSION = "output_dimension"
CURRENT_SERVER_ROUND = "current_server_round"

from typing import Dict, Union

from flwr.common.typing import Scalar
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer, TfidfTransformer, TfidfVectorizer

TextFeatureTransformer = Union[CountVectorizer, TfidfTransformer, TfidfVectorizer, HashingVectorizer]

# Constants that specify the data type of tabular data columns.
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

# The default fill-in values for entire missing columns of a certain data type.
DEFAULT_FILL_VALUES: Dict[str, Scalar] = {BINARY: 0, NUMERIC: 0.0, ORDINAL: UNKNOWN, STRING: "N/A"}

# constants used in config for communication between
# the server and clients.

# FORMAT_SPECIFIED indicates whether the server has the "source of truth"
# to be used for feature alignment.
FORMAT_SPECIFIED = "format_specified"

# FEATURE_INFO refers to the encoded feature information (source of truth).
FEATURE_INFO = "feature_info"

# These are used to initialize a global model on the server's side.
INPUT_DIMENSION = "input_dimension"
OUTPUT_DIMENSION = "output_dimension"

CURRENT_SERVER_ROUND = "current_server_round"

from typing import Dict, Union

from flwr.common.typing import Scalar
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer, TfidfTransformer, TfidfVectorizer

TextFeatureTransformer = Union[CountVectorizer, TfidfTransformer, TfidfVectorizer, HashingVectorizer]


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

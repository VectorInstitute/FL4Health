from enum import Enum

from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer, TfidfTransformer, TfidfVectorizer


TextFeatureTransformer = CountVectorizer | TfidfTransformer | TfidfVectorizer | HashingVectorizer

# constants used in config for communication between
# the server and clients.

# FORMAT_SPECIFIED indicates whether the server has the "source of truth"
# to be used for feature alignment.
SOURCE_SPECIFIED = "source_specified"

# FEATURE_INFO refers to the encoded feature information (source of truth).
FEATURE_INFO = "feature_info"

# These are used to initialize a global model on the server's side.
INPUT_DIMENSION = "input_dimension"
OUTPUT_DIMENSION = "output_dimension"

CURRENT_SERVER_ROUND = "current_server_round"


class FeatureType(Enum):
    NUMERIC = "numeric"
    BINARY = "binary"
    STRING = "string"
    ORDINAL = "ordinal"
    CATEGORICAL_INDICATOR = "categorical_indicator"


FEATURE_TYPES = [
    FeatureType.NUMERIC,
    FeatureType.BINARY,
    FeatureType.STRING,
    FeatureType.ORDINAL,
]

FEATURE_INDICATOR_ATTR = "indicator_of"
FEATURE_MAPPING_ATTR = "mapping"
FEATURE_TYPE_ATTR = "type_"
FEATURE_TARGET_ATTR = "target"

FEATURE_META_ATTR_DEFAULTS = {
    FEATURE_TARGET_ATTR: False,
    FEATURE_INDICATOR_ATTR: None,
    FEATURE_MAPPING_ATTR: None,
}

FEATURE_META_ATTRS = [
    FEATURE_TYPE_ATTR,
    FEATURE_TARGET_ATTR,
    FEATURE_INDICATOR_ATTR,
    FEATURE_MAPPING_ATTR,
]

MISSING_CATEGORY = "null_category"

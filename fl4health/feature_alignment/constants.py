from typing import Union

from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer, TfidfTransformer, TfidfVectorizer

TextFeatureTransformer = Union[CountVectorizer, TfidfTransformer, TfidfVectorizer, HashingVectorizer]

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

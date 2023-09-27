from typing import Union

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

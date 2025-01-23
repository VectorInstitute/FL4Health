from enum import Enum


class LoggingMode(str, Enum):
    TRAIN = "Training"
    EARLY_STOP_VALIDATION = "Early_Stop_Validation"
    VALIDATION = "Validation"
    TEST = "Testing"

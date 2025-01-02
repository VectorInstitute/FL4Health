from enum import Enum


class LoggingMode(Enum):
    TRAIN = "Training"
    EARLY_STOP_VALIDATION = "Early_Stop_Validation"
    VALIDATION = "Validation"
    TEST = "Testing"

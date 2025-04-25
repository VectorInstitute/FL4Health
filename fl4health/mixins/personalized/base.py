"""Base Personalized Mixins"""

from enum import Enum


class PersonalizedMethod(str, Enum):
    DITTO = "ditto"


class BasePersonalizedMixin:
    """A Mixin for transforming an FL Client to a personalized one.

    This mixin is used for validations.
    """

    pass

"""Base Personalized Mixins"""

from abc import ABC, abstractmethod
from enum import Enum

from fl4health.clients.basic_client import BasicClient


class PersonalizedMethod(str, Enum):
    DITTO = "ditto"


class BasePersonalizedMixin:
    """A Mixin for transforming an FL Client to a personalized one.

    This mixin is used for validations.
    """

    pass

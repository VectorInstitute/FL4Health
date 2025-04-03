"""Base Personalized Mixins"""

from enum import Enum
from fl4health.clients.basic_client import BasicClient
from abc import ABC, abstractmethod


class PersonalizedMethod(str, Enum):
    DITTO = "ditto"


class BasePersonalizedMixin(ABC):
    """A Mixin for transforming an FL Client to a personalized one."""

    @abstractmethod
    def to_personalized(self, method: PersonalizedMethod) -> BasicClient:
        """Returns a personalized client."""
        raise NotImplementedError

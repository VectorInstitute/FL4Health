from .ditto import DittoPersonalizedMixin
from fl4health.clients.basic_client import BasicClient

from abc import ABC, abstractmethod
from enum import Enum

from fl4health.clients.basic_client import BasicClient


class PersonalizedModes(str, Enum):
    DITTO = "ditto"


PersonalizedMixinRegistry = {"ditto": DittoPersonalizedMixin}


def make_it_personal(client_base_type: type[BasicClient], mode: PersonalizedModes) -> BasicClient:
    if mode == "ditto":

        return type(
            f"Ditto{client_base_type.__name__}",
            (
                PersonalizedMixinRegistry[mode],
                client_base_type,
            ),
            {},
        )
    else:
        raise ValueError("Unrecognized personalized mode.")

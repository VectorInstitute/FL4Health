from enum import Enum

from fl4health.clients.basic_client import BasicClient
from fl4health.mixins.personalized.ditto import DittoPersonalizedMixin, DittoPersonalizedProtocol


class PersonalizedMode(Enum):
    DITTO = "ditto"


PersonalizedMixinRegistry = {PersonalizedMode.DITTO: DittoPersonalizedMixin}


def make_it_personal(client_base_type: type[BasicClient], mode: PersonalizedMode) -> type[BasicClient]:
    """A mixed class factory for converting basic clients to personalized versions."""
    if mode == PersonalizedMode.DITTO:

        return type(
            f"Ditto{client_base_type.__name__}",
            (
                PersonalizedMixinRegistry[mode],
                client_base_type,
            ),
            {
                # Special flag to bypass validation
                "_dynamically_created": True
            },
        )
    else:
        raise ValueError("Unrecognized personalized mode.")


__all__ = [
    "DittoPersonalizedMixin",
    "DittoPersonalizedProtocol",
    "PersonalizedMode",
    "PersonalizedMixinRegistry",
    "make_it_personal",
]

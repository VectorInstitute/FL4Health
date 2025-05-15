from enum import Enum

from fl4health.clients.basic_client import BasicClient
from fl4health.mixins.personalized.ditto import DittoPersonalizedMixin, DittoPersonalizedProtocol


class PersonalizedModes(str, Enum):
    DITTO = "ditto"


PersonalizedMixinRegistry = {"ditto": DittoPersonalizedMixin}


def make_it_personal(client_base_type: type[BasicClient], mode: PersonalizedModes | str) -> type[BasicClient]:
    """A mixed class factory for converting basic clients to personalized versions."""
    if mode == "ditto":

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
    "PersonalizedModes",
    "PersonalizedMixinRegistry",
    "make_it_personal",
]

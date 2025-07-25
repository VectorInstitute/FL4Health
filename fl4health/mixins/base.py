"""BaseFlexibleMixin."""

from logging import ERROR
from typing import Any

from flwr.common.logger import log

from fl4health.clients.flexible.base import FlexibleClient


class BaseFlexibleMixin:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize a base mixin."""
        super().__init__(*args, **kwargs)

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """This method is called when a class inherits from BaseFlexibleMixin."""
        super().__init_subclass__(**kwargs)

        # Skip check for other mixins
        if cls.__name__.endswith("Mixin"):
            return

        # Skip validation for dynamically created classes
        if hasattr(cls, "_dynamically_created"):
            return

        # Check at class definition time if the parent class satisfies FlexibleClientProtocol
        for base in cls.__bases__:
            if base is not BaseFlexibleMixin and issubclass(base, FlexibleClient):
                return

        # If we get here, no compatible base was found
        msg = (
            f"Class {cls.__name__} inherits from BaseFlexibleMixin but none of its other "
            "base classes implement FlexibleClient."
        )
        log(ERROR, msg)
        raise RuntimeError(msg)

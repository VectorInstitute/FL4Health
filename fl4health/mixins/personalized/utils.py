from collections.abc import Callable
from typing import Any

import wrapt

from fl4health.clients.flexible.base import FlexibleClient


@wrapt.decorator
def ensure_protocol_compliance(func: Callable, instance: Any | None, args: Any, kwargs: Any) -> Any:
    """
    Wrapper to ensure that the instance is of ``FlexibleClient`` type.

    **NOTE**: This should only be used within a ``FlexibleClient``.

    Args:
        func (Callable): The function to be wrapped.
        instance (Any | None): The associated instance if it is a method belonging to a class or a standalone
        args (Any): args passed to func.
        kwargs (Any): kwargs passed to func.

    Raises:
        TypeError: We raise this error if the instance is not a ``FlexibleClient``.

    Returns:
        (Any): Application of the function to the args and kwargs.
    """
    # validate self is a FlexibleClient
    if not isinstance(instance, FlexibleClient):
        raise TypeError("Protocol requirements not met.")

    return func(*args, **kwargs)

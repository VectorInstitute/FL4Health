from collections.abc import Callable
from typing import Any

import wrapt

from fl4health.clients.flexible_client import FlexibleClient


@wrapt.decorator
def ensure_protocol_compliance(func: Callable, instance: Any | None, args: Any, kwargs: Any) -> Any:
    """
    Wrapper to ensure that a the instance is of `FlexibleClient` type.

    NOTE: this should only be used within a `FlexibleClient`.

    Args:
        # are params specified and supplied by the `wrapt` decorator
        func (Callable): the function to be wrapped
        instance (Any | None): the associated instance if it is a method belonging to a class or a standalone
        args (Any): args passed to func
        kwargs (Any): kwargs passed to func

    Raises:
        TypeError: we raise error if the instance is not a `FlexibleClient`.

    Returns:
        _type_: _description_
    """
    # validate self is a FlexibleClient
    if not isinstance(instance, FlexibleClient):
        raise TypeError("Protocol requirements not met.")

    return func(*args, **kwargs)

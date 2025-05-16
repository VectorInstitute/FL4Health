from collections.abc import Callable
from typing import Any

import wrapt

from fl4health.clients.basic_client import BasicClient


@wrapt.decorator
def ensure_protocol_compliance(func: Callable, instance: Any | None, args: Any, kwargs: Any) -> Any:
    """Wrapper to ensure that a the instance is of `BasicClient` type.

    Args:
        # are params specified and supplied by the `wrapt` decorator
        func (Callable): the function to be wrapped
        instance (Any | None): the associated instance if it is a method belonging to a class or a standalone
        args (Any): args passed to func
        kwargs (Any): kwargs passed to func

    Raises:
        TypeError: we raise error if the instance is not a `BasicClient`.

    Returns:
        _type_: _description_
    """
    # validate self is a BasicClient
    if not isinstance(instance, BasicClient):
        raise TypeError("Protocol requirements not met.")

    return func(*args, **kwargs)

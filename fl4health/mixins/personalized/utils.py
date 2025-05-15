from collections.abc import Callable
from typing import Any

import wrapt

from fl4health.clients.basic_client import BasicClient


@wrapt.decorator
def ensure_protocol_compliance(func: Callable, instance: Any | None, args: Any, kwargs: Any) -> None:
    # validate self is a BasicClient
    self = instance
    if not isinstance(self, BasicClient):
        raise TypeError("Protocol requirements not met.")

    return func(*args, **kwargs)

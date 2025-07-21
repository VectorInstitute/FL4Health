"""MR MTL Personalized Mixin."""

import warnings
from logging import WARN
from typing import Any, Protocol, runtime_checkable

from flwr.common.logger import log

from fl4health.clients.flexible.base import FlexibleClient
from fl4health.mixins.adaptive_drift_constrained import (
    AdaptiveDriftConstrainedMixin,
    AdaptiveDriftConstrainedProtocol,
)


@runtime_checkable
class MrMtlPersonalizedProtocol(AdaptiveDriftConstrainedProtocol, Protocol):
    pass


class MrMtlPersonalizedMixin(AdaptiveDriftConstrainedMixin):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """
        This client implements the MR-MTL algorithm from MR-MTL: On Privacy and Personalization in Cross-Silo
        Federated Learning. The idea is that we want to train personalized versions of the global model for each
        client. However, instead of using a separate solver for the global model, as in Ditto, we update the initial
        global model with aggregated local models on the server-side and use those weights to also constrain the
        training of a local model. The constraint for this local model is identical to the FedProx loss. The key
        difference is that the local model is never replaced with aggregated weights. It is always local.

        **NOTE**: lambda, the drift loss weight, is initially set and potentially adapted by the server akin to the
        heuristic suggested in the original FedProx paper. Adaptation is optional and can be disabled in the
        corresponding strategy used by the server
        """

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """This method is called when a class inherits from MrMtlPersonalizedMixin."""
        super().__init_subclass__(**kwargs)

        # Skip check for other mixins
        if cls.__name__.endswith("Mixin"):
            return

        # Skip validation for dynamically created classes
        if hasattr(cls, "_dynamically_created"):
            return

        # Check at class definition time if the parent class satisfies FlexibleClientProtocol
        for base in cls.__bases__:
            if base is not MrMtlPersonalizedMixin and issubclass(base, FlexibleClient):
                return

        # If we get here, no compatible base was found
        msg = (
            f"Class {cls.__name__} inherits from MrMtlPersonalizedMixin but none of its other "
            f"base classes implement FlexibleClient. This may cause runtime errors."
        )
        log(WARN, msg)
        warnings.warn(msg, RuntimeWarning, stacklevel=2)

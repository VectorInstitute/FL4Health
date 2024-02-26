from logging import DEBUG
from typing import Any, Optional

from flwr.common.logger import log
from flwr.server.history import History

from fl4health.server.base_server import FlServer


class FedDGGA(FlServer):
    def __init__(
        self,
        server: FlServer,
    ) -> None:
        # TODO docstrings
        FlServer.__init__(
            self,
            client_manager=server.client_manager(),
            strategy=server.strategy,
            wandb_reporter=server.wandb_reporter,
            checkpointer=server.checkpointer,
        )
        self.server = server

    def __getattribute__(self, attr: str) -> Any:
        """
        Redirect all method calls to self.server methods.

        Args:
            attr: the method name

        Returns: the method's return
        """
        # if the user is looking to access a method from this class, returns it
        if attr in type(self).__dict__.keys():
            log(DEBUG, f"Using FedDGGA's own implementation of method or attribute: {attr}")
            return object.__getattribute__(self, attr)

        # getting self.server while avoiding infinite recursion on __getattribute__
        self_server = object.__getattribute__(self, "server")

        # if the user is looking to access self.server, returns it
        if attr == "server":
            return self_server

        if not hasattr(self_server, attr):
            log(DEBUG, f"self.server does not implement method or attribute, using FedDGGA's: {attr}")
            return object.__getattribute__(self, attr)

        log(DEBUG, f"Redirecting method or attribute call from FedDGGA to its self.server: {attr}")
        return getattr(self_server, attr)

    def fit(self, num_rounds: int, timeout: Optional[float]) -> History:
        return self.server.fit(num_rounds, timeout)

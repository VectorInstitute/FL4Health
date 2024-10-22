"""Base Class for Reporters.

Super simple for now but keeping it in a seperate file in case we add more base methods.
"""

from typing import Any


class BaseReporter:
    def report(
        self,
        data: dict,
        round: int | None = None,
        epoch: int | None = None,
        step: int | None = None,
    ) -> None:
        """A method called by clients or servers to send data to the reporter.

        The report method is called by the client/server at frequent intervals (ie step, epoch, round) and sometimes
        outside of a FL round (for high level summary data). It is up to the reporter to determine when and what to
        report.

        Args:
            data (dict): The data to maybe report from the server or client.
            round (int | None, optional): The current FL round. If None, this indicates that the method was called
                outside of a round (e.g. for summary information). Defaults to None.
            epoch (int | None, optional): The current epoch. If None then this method was not called at or within the
                scope of an epoch. Defaults to None.
            step (int | None, optional): The current step (total). If None then this method was called outside the
                scope of a training or evaluation step (eg. at the end of an epoch or round) Defaults to None.
        """
        raise NotImplementedError

    def initialize(self, **kwargs: Any) -> None:
        """Method for initializing reporters with client/server information

        This method is called once by the client or server during initialization.

        Args:
            kwargs (Any): arbitrary keyword arguments containing information from the
                client or server that might be useful for initializing the reporter.
                This information should be treated as optional and this method should
                work even if no keyword arguments are passed.
        """
        pass

    def shutdown(self) -> None:
        """Called by the client/server on shutdown."""
        pass

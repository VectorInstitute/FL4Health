from pathlib import Path
from unittest import mock

from fl4health.reporting.fl_wandb import ClientWandBReporter, ServerWandBReporter


def test_server_wandb_reporter(tmp_path: Path) -> None:
    with mock.patch.object(ServerWandBReporter, "__init__", lambda a, b, c, d, e, f, g, h: None):
        reporter = ServerWandBReporter("", "", "", "", None, None, {})
        log_dir = str(tmp_path.joinpath("fl_wandb_logs"))
        reporter._maybe_create_local_log_directory(log_dir)
        assert log_dir in list(map(lambda x: str(x), tmp_path.iterdir()))


def test_client_wandb_reporter(tmp_path: Path) -> None:
    with mock.patch.object(ClientWandBReporter, "__init__", lambda a, b, c, d, e: None):
        reporter = ClientWandBReporter("", "", "", "")
        log_dir = str(tmp_path.joinpath("fl_wandb_logs"))
        reporter._maybe_create_local_log_directory(log_dir)
        assert log_dir in list(map(lambda x: str(x), tmp_path.iterdir()))

# keep for backwards compatibility
from logging import WARNING

from flwr.common.logger import log

from fl4health.metrics import *  # noqa: F401, F403


log(WARNING, "Metrics now reside at fl4health/metrics/metrics.py. This path will be removed in future releases.")

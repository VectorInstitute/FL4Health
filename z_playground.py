# from flamby.datasets.fed_isic2019 import BATCH_SIZE, LR, NUM_CLIENTS, Baseline, BaselineLoss
from flamby.datasets.fed_heart_disease import Baseline
from fl4health.server.secure_aggregation_utils import get_model_layer_types

net = Baseline()

print(get_model_layer_types(net))
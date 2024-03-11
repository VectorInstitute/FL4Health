# from flamby.datasets.fed_heart_disease import BATCH_SIZE, LR, NUM_CLIENTS, Baseline, BaselineLoss
# from research.flamby_distributed_dp.fed_ixi.model import ModifiedBaseline
from research.flamby_distributed_dp.fed_isic2019.model import ModifiedBaseline


print(sum(p.numel() for p in ModifiedBaseline().parameters()))
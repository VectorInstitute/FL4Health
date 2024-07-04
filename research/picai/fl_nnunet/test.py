import torch

_ = torch.manual_seed(0)
from torchmetrics.segmentation import GeneralizedDiceScore, MeanIoU

gds = GeneralizedDiceScore(num_classes=2)
miou = MeanIoU(num_classes=2)
preds = torch.randint(0, 2, (10, 2, 128, 128))
target = torch.randint(0, 2, (10, 2, 128, 128))
print(gds(preds, target))
print(miou(preds, target))

gds = GeneralizedDiceScore(num_classes=2, per_class=True)
miou = MeanIoU(num_classes=2, per_class=True)
print(gds(preds, target))
print(miou(preds, target))

gds = GeneralizedDiceScore(num_classes=2, per_class=True, include_background=False)
miou = MeanIoU(num_classes=2, per_class=True, include_background=False)
print(gds(preds, target))
print(miou(preds, target))

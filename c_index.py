from fl4health.utils.metrics import C_Index
import torch 

meter = C_Index()

preds = torch.tensor([[0.7386],
        [0.8290],
        [0.4809],
        [0.2047],
        [0.5662]])

target = torch.tensor([[   0.,  598.],
        [   0., 1009.],
        [   0.,  586.],
        [   0.,  323.],
        [   0.,  216.]])
metric = meter(preds=0*preds,target=target)
print(metric)

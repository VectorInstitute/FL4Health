import torch 
from research.flamby.flamby_data_utils import construct_fedisic_train_val_datasets
from torch.utils.data import DataLoader
from opacus import PrivacyEngine


train_dataset, validation_dataset = construct_fedisic_train_val_datasets(1, 'flamby_datasets/fed_isic2019')
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, generator=torch.Generator(device='cuda' if torch.cuda.is_available() else "cpu"))
val_loader = DataLoader(validation_dataset, batch_size=1, shuffle=False, generator=torch.Generator(device='cuda' if torch.cuda.is_available() else "cpu"))

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 1)
    def forward(self, x):
        return self.linear(x)


model = Model()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

m, o, t = PrivacyEngine().make_private(
    module=model,
    optimizer=optimizer,
    data_loader=train_loader,
    noise_multiplier=0.1,
    max_grad_norm=20,
    clipping="flat",
    poisson_sampling=False
)

i = 0
for batches in iter(train_loader):
    i += 1
    if i % 10 == 0:
        print('batches',batches[0].shape, batches[1])
    if i == 100:
        break

i = 0
for batches in iter(t):
    i += 1
    if i % 10 == 0:
        print('opacus', batches[0].shape, batches[1])
    # if torch.numerl(batches[1]) == 0:
    #     print('empty')
    if i == 100:
        break
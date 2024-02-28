# import torch
# import torch.nn as nn

# # Define a simple model
# class Model(nn.Module):
#     def __init__(self):
#         super(Model, self).__init__()
#         self.fc = nn.Linear(10, 1)
    
#     def forward(self, x):
#         return self.fc(x)

# # Instantiate the model
# model = Model()

# # Create some dummy input and target
# input_data = torch.randn(1, 10)
# target = torch.randn(1, 1)

# # Forward pass
# output = model(input_data)

# # Compute loss
# criterion = nn.MSELoss()
# loss = criterion(output, target)

# # Backward pass
# loss.backward()

# # Access gradients
# gradients = [param.grad for param in model.parameters() if param.grad is not None]
# # Print gradients
# for grad in gradients:
#     print(grad)

# for param in model.parameters():
#     param.grad *= 1_000

# # Access gradients
# gradients = [param.grad for param in model.parameters() if param.grad is not None]
# # Print gradients
# for grad in gradients:
#     print(grad)


from fl4health.privacy_mechanisms.discrete_gaussian_mechanism import modular_clipping
import numpy as np
modulus = 6
print(modular_clipping(np.array([-10,9,8,5,3,32]), -3, 3))
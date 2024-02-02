import math
import torch
from sympy import fwht

#x = torch.tensor([1.0, 0, 1, 0, 0, 1, 1, 0])
x = torch.rand(256)
xt1 = fwht(x)
#print("sympy: ", xt1) 

dim=x.size()[0]
log2 = int(math.ceil(math.log2(dim)))
h_2x2 = torch.tensor([[1.0, 1.0], [1.0, -1.0]])
permutation = torch.tensor([0,2,1])

def _hadamard_step(x, dim):
    x_shape = x.size()
    x = x.reshape(-1, 2)
    #print(x)
    x = torch.matmul(x, h_2x2)
    x = x.view(-1, x_shape[0] // 2, 2)
    x = torch.transpose(x,2,1)
    x = x.reshape(x_shape)
    #print(x)
    return x 

#x = x.reshape(-1, 2, dim // 2)
index = torch.tensor(0)
def cond(i, x):
    return i < log2

def body(i, x):
    return i + 1, _hadamard_step(x, dim)

while cond(index,log2):
    index,x = body(index,x)
    
xt2 = x.view(-1, dim)
xt2 = xt2.tolist()[0]
#print("pytorch: ", xt2)

diff = sum([x1-x2 for x1, x2 in zip(xt1,xt2)])
print(diff)

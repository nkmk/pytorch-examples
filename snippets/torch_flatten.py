import torch
import torch.nn as nn

print(torch.__version__)
# 1.7.1

t = torch.zeros(2, 3, 4, 5)
print(t.shape)
# torch.Size([2, 3, 4, 5])

print(torch.flatten(t).shape)
# torch.Size([120])

print(torch.flatten(t, 1, 2).shape)
# torch.Size([2, 12, 5])

flatten = nn.Flatten()
print(flatten(t).shape)
# torch.Size([2, 60])

flatten_all = nn.Flatten(0, -1)
print(flatten_all(t).shape)
# torch.Size([120])

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.flatten(x)
        return x

net = Net()
print(net(t).shape)
# torch.Size([2, 60])

class NetFunctional(nn.Module):
    def forward(self, x):
        x = torch.flatten(x)
        return x

net_func = NetFunctional()
print(net_func(t).shape)
# torch.Size([120])

class NetFunctionalDim(nn.Module):
    def forward(self, x):
        x = torch.flatten(x, 1)
        return x

net_func_dim = NetFunctionalDim()
print(net_func_dim(t).shape)
# torch.Size([2, 60])

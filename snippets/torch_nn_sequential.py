import torch
import torch.nn as nn

print(torch.__version__)
# 1.7.1

torch.manual_seed(0)

net_seq = nn.Sequential(
    nn.Linear(1000, 100),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(100, 10)
)

print(net_seq)
# Sequential(
#   (0): Linear(in_features=1000, out_features=100, bias=True)
#   (1): ReLU()
#   (2): Dropout(p=0.2, inplace=False)
#   (3): Linear(in_features=100, out_features=10, bias=True)
# )

print(type(net_seq))
# <class 'torch.nn.modules.container.Sequential'>

print(issubclass(type(net_seq), nn.Module))
# True

print(net_seq[0])
# Linear(in_features=1000, out_features=100, bias=True)

print(type(net_seq[0]))
# <class 'torch.nn.modules.linear.Linear'>

print(net_seq[0].weight)
# Parameter containing:
# tensor([[-0.0002,  0.0170, -0.0260,  ..., -0.0102,  0.0001, -0.0061],
#         [-0.0027, -0.0247, -0.0002,  ...,  0.0012, -0.0096,  0.0238],
#         [ 0.0175,  0.0057,  0.0048,  ..., -0.0144, -0.0125, -0.0265],
#         ...,
#         [ 0.0007,  0.0006, -0.0082,  ..., -0.0033, -0.0160, -0.0130],
#         [ 0.0016, -0.0262,  0.0075,  ...,  0.0072,  0.0184,  0.0094],
#         [ 0.0031,  0.0199, -0.0057,  ..., -0.0101, -0.0229, -0.0243]],
#        requires_grad=True)

torch.manual_seed(0)
t = torch.randn(1, 1000)

torch.manual_seed(0)
print(net_seq(t))
# tensor([[-0.3884,  0.0370,  0.0175,  0.3579,  0.1390, -0.4750, -0.3484,  0.2648,
#           0.1452,  0.1219]], grad_fn=<AddmmBackward>)

print(net_seq(t).shape)
# torch.Size([1, 10])

t_ = torch.randn(3, 1000)
print(net_seq(t_))
# tensor([[-0.4004, -0.1475,  0.0014, -0.0756,  0.2095, -0.3645,  0.7861, -0.0645,
#           0.1356, -0.0600],
#         [-0.2170, -0.0610,  0.0520, -0.0137,  0.1295,  0.0086,  0.0625, -0.6118,
#           0.1942, -0.5471],
#         [-0.2405, -0.0499, -0.1613,  0.4955,  0.1280, -0.3260, -0.1218, -0.1814,
#           0.1854,  0.0027]], grad_fn=<AddmmBackward>)

print(net_seq(t_).shape)
# torch.Size([3, 10])

from collections import OrderedDict

torch.manual_seed(0)

net_seq_od = nn.Sequential(OrderedDict([
    ('fc1', nn.Linear(1000, 100)),
    ('relu', nn.ReLU()),
    ('dropout', nn.Dropout(0.2)),
    ('fc2', nn.Linear(100, 10))
]))

print(net_seq_od)
# Sequential(
#   (fc1): Linear(in_features=1000, out_features=100, bias=True)
#   (relu): ReLU()
#   (dropout): Dropout(p=0.2, inplace=False)
#   (fc2): Linear(in_features=100, out_features=10, bias=True)
# )

torch.manual_seed(0)
print(net_seq_od(t))
# tensor([[-0.3884,  0.0370,  0.0175,  0.3579,  0.1390, -0.4750, -0.3484,  0.2648,
#           0.1452,  0.1219]], grad_fn=<AddmmBackward>)

print(net_seq_od[0])
# Linear(in_features=1000, out_features=100, bias=True)

print(net_seq_od.fc1)
# Linear(in_features=1000, out_features=100, bias=True)

# print(net_seq_od['fc1'])
# TypeError: 'str' object cannot be interpreted as an integer

torch.manual_seed(0)

net_seq_add = nn.Sequential()
net_seq_add.add_module('fc1', nn.Linear(1000, 100))
net_seq_add.add_module('relu', nn.ReLU())
net_seq_add.add_module('dropout', nn.Dropout(0.2))
net_seq_add.add_module('fc2', nn.Linear(100, 10))

print(net_seq_add)
# Sequential(
#   (fc1): Linear(in_features=1000, out_features=100, bias=True)
#   (relu): ReLU()
#   (dropout): Dropout(p=0.2, inplace=False)
#   (fc2): Linear(in_features=100, out_features=10, bias=True)
# )

torch.manual_seed(0)
print(net_seq_add(t))
# tensor([[-0.3884,  0.0370,  0.0175,  0.3579,  0.1390, -0.4750, -0.3484,  0.2648,
#           0.1452,  0.1219]], grad_fn=<AddmmBackward>)

print(net_seq_add[0])
# Linear(in_features=1000, out_features=100, bias=True)

print(net_seq_add.fc1)
# Linear(in_features=1000, out_features=100, bias=True)

import torch
import torch.nn as nn
import torch.nn.functional as F

print(torch.__version__)
# 1.7.1

class NetInner(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(1000, 100)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = F.relu(self.fc(x))
        x = self.dropout(x)
        return x

torch.manual_seed(0)

net_nested_seq = nn.Sequential(
    NetInner(),
    nn.Linear(100, 10)
)

print(net_nested_seq)
# Sequential(
#   (0): NetInner(
#     (fc): Linear(in_features=1000, out_features=100, bias=True)
#     (dropout): Dropout(p=0.2, inplace=False)
#   )
#   (1): Linear(in_features=100, out_features=10, bias=True)
# )

print(net_nested_seq[0])
# NetInner(
#   (fc): Linear(in_features=1000, out_features=100, bias=True)
#   (dropout): Dropout(p=0.2, inplace=False)
# )

print(net_nested_seq[0].fc)
# Linear(in_features=1000, out_features=100, bias=True)

torch.manual_seed(0)
t = torch.randn(1, 1000)

torch.manual_seed(0)
print(net_nested_seq(t))
# tensor([[-0.3884,  0.0370,  0.0175,  0.3579,  0.1390, -0.4750, -0.3484,  0.2648,
#           0.1452,  0.1219]], grad_fn=<AddmmBackward>)

class NetNested(nn.Module):
    def __init__(self):
        super().__init__()
        self.my_net = NetInner()
        self.fc = nn.Linear(100, 10)

    def forward(self, x):
        x = self.my_net(x)
        x = self.fc(x)
        return x

torch.manual_seed(0)
net_nested = NetNested()
print(net_nested)
# NetNested(
#   (my_net): NetInner(
#     (fc): Linear(in_features=1000, out_features=100, bias=True)
#     (dropout): Dropout(p=0.2, inplace=False)
#   )
#   (fc): Linear(in_features=100, out_features=10, bias=True)
# )

print(net_nested.my_net)
# NetInner(
#   (fc): Linear(in_features=1000, out_features=100, bias=True)
#   (dropout): Dropout(p=0.2, inplace=False)
# )

print(net_nested.my_net.fc)
# Linear(in_features=1000, out_features=100, bias=True)

torch.manual_seed(0)
print(net_nested(t))
# tensor([[-0.3884,  0.0370,  0.0175,  0.3579,  0.1390, -0.4750, -0.3484,  0.2648,
#           0.1452,  0.1219]], grad_fn=<AddmmBackward>)

class NetNestedSeq(nn.Module):
    def __init__(self):
        super().__init__()
        self.my_net = nn.Sequential(
            nn.Linear(1000, 100),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.fc = nn.Linear(100, 10)

    def forward(self, x):
        x = self.my_net(x)
        x = self.fc(x)
        return x

torch.manual_seed(0)
net_nested_seq = NetNestedSeq()
print(net_nested_seq)
# NetNestedSeq(
#   (my_net): Sequential(
#     (0): Linear(in_features=1000, out_features=100, bias=True)
#     (1): ReLU()
#     (2): Dropout(p=0.2, inplace=False)
#   )
#   (fc): Linear(in_features=100, out_features=10, bias=True)
# )

print(net_nested_seq.my_net)
# Sequential(
#   (0): Linear(in_features=1000, out_features=100, bias=True)
#   (1): ReLU()
#   (2): Dropout(p=0.2, inplace=False)
# )

print(net_nested_seq.my_net[0])
# Linear(in_features=1000, out_features=100, bias=True)

torch.manual_seed(0)
print(net_nested_seq(t))
# tensor([[-0.3884,  0.0370,  0.0175,  0.3579,  0.1390, -0.4750, -0.3484,  0.2648,
#           0.1452,  0.1219]], grad_fn=<AddmmBackward>)

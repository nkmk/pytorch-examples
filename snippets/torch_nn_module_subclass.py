import torch
import torch.nn as nn
import torch.nn.functional as F

print(torch.__version__)
# 1.7.1

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1000, 100)
        self.fc2 = nn.Linear(100, 10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

#     def forward(self, x):
#         x = self.relu(self.fc1(x))
#         x = self.dropout(x)
#         x = self.fc2(x)
#         return x

#     def forward(self, x):
#         return self.fc2(self.dropout(self.relu(self.fc1(x))))

torch.manual_seed(0)
net = Net()
print(net)
# Net(
#   (fc1): Linear(in_features=1000, out_features=100, bias=True)
#   (fc2): Linear(in_features=100, out_features=10, bias=True)
#   (relu): ReLU()
#   (dropout): Dropout(p=0.2, inplace=False)
# )

torch.manual_seed(0)
t = torch.randn(1, 1000)

torch.manual_seed(0)
print(net(t))
# tensor([[-0.3884,  0.0370,  0.0175,  0.3579,  0.1390, -0.4750, -0.3484,  0.2648,
#           0.1452,  0.1219]], grad_fn=<AddmmBackward>)

# print(net[0])
# TypeError: 'Net' object is not subscriptable

print(net.fc1)
# Linear(in_features=1000, out_features=100, bias=True)

# print(net['fc1'])
# TypeError: 'Net' object is not subscriptable

class NetParam(nn.Module):
    def __init__(self, n_input, n_hidden, n_output, p_dropout):
        super().__init__()
        self.fc1 = nn.Linear(n_input, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_output)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p_dropout)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

torch.manual_seed(0)
net_param = NetParam(1000, 100, 10, 0.2)
print(net_param)
# NetParam(
#   (fc1): Linear(in_features=1000, out_features=100, bias=True)
#   (fc2): Linear(in_features=100, out_features=10, bias=True)
#   (relu): ReLU()
#   (dropout): Dropout(p=0.2, inplace=False)
# )

torch.manual_seed(0)
print(net_param(t))
# tensor([[-0.3884,  0.0370,  0.0175,  0.3579,  0.1390, -0.4750, -0.3484,  0.2648,
#           0.1452,  0.1219]], grad_fn=<AddmmBackward>)

class NetFunctional(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1000, 100)
        self.fc2 = nn.Linear(100, 10)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

torch.manual_seed(0)
net_f = NetFunctional()
print(net_f)
# NetFunctional(
#   (fc1): Linear(in_features=1000, out_features=100, bias=True)
#   (fc2): Linear(in_features=100, out_features=10, bias=True)
#   (dropout): Dropout(p=0.2, inplace=False)
# )

torch.manual_seed(0)
print(net_f(t))
# tensor([[-0.3884,  0.0370,  0.0175,  0.3579,  0.1390, -0.4750, -0.3484,  0.2648,
#           0.1452,  0.1219]], grad_fn=<AddmmBackward>)

torch.manual_seed(0)
net = Net()

net.train()

torch.manual_seed(0)
print(net(t))
# tensor([[-0.3884,  0.0370,  0.0175,  0.3579,  0.1390, -0.4750, -0.3484,  0.2648,
#           0.1452,  0.1219]], grad_fn=<AddmmBackward>)

net.eval()

torch.manual_seed(0)
print(net(t))
# tensor([[-0.2834,  0.0206, -0.0293,  0.3862,  0.3254, -0.5541, -0.1213,  0.1510,
#          -0.0269, -0.0560]], grad_fn=<AddmmBackward>)

class NetFunctionalDropoutError(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1000, 100)
        self.fc2 = nn.Linear(100, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.dropout(x, 0.2)
        x = self.fc2(x)
        return x

torch.manual_seed(0)
net_f_dropout_error = NetFunctionalDropoutError()

net_f_dropout_error.train()

torch.manual_seed(0)
print(net_f_dropout_error(t))
# tensor([[-0.3884,  0.0370,  0.0175,  0.3579,  0.1390, -0.4750, -0.3484,  0.2648,
#           0.1452,  0.1219]], grad_fn=<AddmmBackward>)

net_f_dropout_error.eval()

torch.manual_seed(0)
print(net_f_dropout_error(t))
# tensor([[-0.3884,  0.0370,  0.0175,  0.3579,  0.1390, -0.4750, -0.3484,  0.2648,
#           0.1452,  0.1219]], grad_fn=<AddmmBackward>)

class NetFunctionalDropout(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1000, 100)
        self.fc2 = nn.Linear(100, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.dropout(x, 0.2, self.training)
        x = self.fc2(x)
        return x

torch.manual_seed(0)
net_f_dropout = NetFunctionalDropout()

net_f_dropout.train()

torch.manual_seed(0)
print(net_f_dropout(t))
# tensor([[-0.3884,  0.0370,  0.0175,  0.3579,  0.1390, -0.4750, -0.3484,  0.2648,
#           0.1452,  0.1219]], grad_fn=<AddmmBackward>)

net_f_dropout.eval()

torch.manual_seed(0)
print(net_f_dropout(t))
# tensor([[-0.2834,  0.0206, -0.0293,  0.3862,  0.3254, -0.5541, -0.1213,  0.1510,
#          -0.0269, -0.0560]], grad_fn=<AddmmBackward>)

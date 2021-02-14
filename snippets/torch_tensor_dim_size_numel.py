import torch

print(torch.__version__)
# 1.7.1

t = torch.zeros(2, 3)
print(t)
# tensor([[0., 0., 0.],
#         [0., 0., 0.]])

print(type(t))
# <class 'torch.Tensor'>

print(t.dim())
# 2

print(type(t.dim()))
# <class 'int'>

print(t.ndimension())
# 2

print(t.ndim)
# 2

print(t.size())
# torch.Size([2, 3])

print(type(t.size()))
# <class 'torch.Size'>

print(issubclass(type(t.size()), tuple))
# True

print(t.size()[1])
# 3

print(type(t.size()[1]))
# <class 'int'>

a, b = t.size()
print(a)
# 2

print(b)
# 3

print(t.shape)
# torch.Size([2, 3])

print(torch.numel(t))
# 6

print(type(torch.numel(t)))
# <class 'int'>

print(t.numel())
# 6

print(t.nelement())
# 6

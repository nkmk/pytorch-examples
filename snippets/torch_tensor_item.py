import torch

print(torch.__version__)
# 1.7.1

t = torch.arange(6).reshape(2, 3)
print(t)
# tensor([[0, 1, 2],
#         [3, 4, 5]])

print(t[1, 1])
# tensor(4)

print(type(t[1, 1]))
# <class 'torch.Tensor'>

print(t[1, 1].ndim)
# 0

print(t[1, 1].item())
# 4

print(type(t[1, 1].item()))
# <class 'int'>

print(t[:2, 1])
# tensor([1, 4])

# print(t[:2, 1].item())
# ValueError: only one element tensors can be converted to Python scalars

print(t[1, [1]])
# tensor([4])

print(t[1, [1]].ndim)
# 1

print(t[1, [1]].item())
# 4

print(int(t[1, 1]))
# 4

print(float(t[1, 1]))
# 4.0

print(str(t[1, 1]))
# tensor(4)

print(type(str(t[1, 1])))
# <class 'str'>

print(str(t[1, 1].item()))
# 4

print(type(str(t[1, 1].item())))
# <class 'str'>

# print(int(t[:2, 1]))
# ValueError: only one element tensors can be converted to Python scalars

print(int(t[1, [1]]))
# 4

# print(float(t[:2, 1]))
# ValueError: only one element tensors can be converted to Python scalars

print(float(t[1, [1]]))
# 4.0

print(str(t[:2, 1]))
# tensor([1, 4])

print(type(str(t[:2, 1])))
# <class 'str'>

print(torch.max(t))
# tensor(5)

print(torch.max(t).item())
# 5

print(torch.sum(t))
# tensor(15)

print(torch.sum(t).item())
# 15

print(torch.sum(t, 0))
# tensor([3, 5, 7])

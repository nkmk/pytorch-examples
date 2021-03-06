import torch

print(torch.__version__)
# 1.7.1

print(type(torch.float32))
# <class 'torch.dtype'>

print(torch.float32 is torch.float)
# True

print(torch.int64 is torch.long)
# True

t_float32 = torch.tensor([0.1, 1.5, 2.9])
print(t_float32)
# tensor([0.1000, 1.5000, 2.9000])

print(t_float32.dtype)
# torch.float32

print(type(t_float32.dtype))
# <class 'torch.dtype'>

t_float64 = torch.tensor([0.1, 1.5, 2.9], dtype=torch.float64)
print(t_float64.dtype)
# torch.float64

t_int32 = torch.ones(3, dtype=torch.int32)
print(t_int32.dtype)
# torch.int32

t_float64 = t_float32.to(torch.float64)
print(t_float64.dtype)
# torch.float64

print(t_float32)
# tensor([0.1000, 1.5000, 2.9000])

print(t_float32.to(torch.int64))
# tensor([0, 1, 2])

t_float64 = t_float32.double()
print(t_float64.dtype)
# torch.float64

# t_float32.float64()
# AttributeError: 'Tensor' object has no attribute 'float64'

t_float16 = torch.ones(3, dtype=torch.float16)
t_int64 = torch.ones(3, dtype=torch.int64)

print((t_float16 + t_int64).dtype)
# torch.float16

t_float32 = torch.ones(3, dtype=torch.float32)
t_float64 = torch.ones(3, dtype=torch.float64)

print((t_float32 + t_float64).dtype)
# torch.float64

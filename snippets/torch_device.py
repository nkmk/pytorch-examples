import torch

print(torch.__version__)
# 1.7.1

print(torch.tensor([0.1, 0.2]))
# tensor([0.1000, 0.2000])

print(torch.tensor([0.1, 0.2], device=torch.device('cpu')))
# tensor([0.1000, 0.2000])

print(torch.tensor([0.1, 0.2], device='cpu'))
# tensor([0.1000, 0.2000])

print(torch.tensor([0.1, 0.2], device=torch.device('cuda:0')))
# tensor([0.1000, 0.2000], device='cuda:0')

print(torch.tensor([0.1, 0.2], device=torch.device('cuda')))
# tensor([0.1000, 0.2000], device='cuda:0')

print(torch.tensor([0.1, 0.2], device=torch.device(0)))
# tensor([0.1000, 0.2000], device='cuda:0')

print(torch.tensor([0.1, 0.2], device='cuda:0'))
# tensor([0.1000, 0.2000], device='cuda:0')

print(torch.tensor([0.1, 0.2], device='cuda'))
# tensor([0.1000, 0.2000], device='cuda:0')

print(torch.tensor([0.1, 0.2], device=0))
# tensor([0.1000, 0.2000], device='cuda:0')

t_cpu = torch.tensor([0.1, 0.2])
print(t_cpu.device)
# cpu

print(type(t_cpu.device))
# <class 'torch.device'>

t_gpu = torch.tensor([0.1, 0.2], device='cuda')
print(t_gpu.device)
# cuda:0

print(type(t_gpu.device))
# <class 'torch.device'>

print(t_cpu.is_cuda)
# False

print(t_gpu.is_cuda)
# True

t_cpu = torch.tensor([0.1, 0.2])
print(t_cpu.device)
# cpu

t_gpu = t_cpu.to('cuda')
print(t_gpu.device)
# cuda:0

print(t_cpu.to('cuda', torch.float64))
# tensor([0.1000, 0.2000], device='cuda:0', dtype=torch.float64)

# print(t_cpu.to(torch.float64, 'cuda'))
# TypeError: to() received an invalid combination of arguments - got (torch.dtype, str), but expected one of:
#  * (torch.device device, torch.dtype dtype, bool non_blocking, bool copy, *, torch.memory_format memory_format)
#  * (torch.dtype dtype, bool non_blocking, bool copy, *, torch.memory_format memory_format)
#  * (Tensor tensor, bool non_blocking, bool copy, *, torch.memory_format memory_format)

print(t_cpu.to(dtype=torch.float64, device='cuda'))
# tensor([0.1000, 0.2000], device='cuda:0', dtype=torch.float64)

print(t_cpu.cuda().device)
# cuda:0

print(t_cpu.cuda(0).device)
# cuda:0

# print(t_cpu.cuda(1).device)
# RuntimeError: CUDA error: invalid device ordinal

print(t_cpu.cuda('cuda:0').device)
# cuda:0

# print(t_cpu.cuda('cpu').device)
# RuntimeError: Invalid device, must be cuda device

print(t_gpu.cpu().device)
# cpu

t_cpu2 = t_cpu.to('cpu')
print(t_cpu is t_cpu2)
# True

t_gpu2 = t_gpu.cuda()
print(t_gpu is t_gpu2)
# True

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
print(device)
# cuda:0

t = torch.tensor([0.1, 0.2], device=device)
print(t.device)
# cuda:0

torch.manual_seed(0)

net = torch.nn.Linear(2, 2)
print(isinstance(net, torch.nn.Module))
# True

print(net.weight.device)
# cpu

net.to('cuda')
print(net.weight.device)
# cuda:0

net.cpu()
print(net.weight.device)
# cpu

net.cuda()
print(net.weight.device)
# cuda:0

t_gpu = torch.tensor([0.1, 0.2], device='cuda')
print(t_gpu.device)
# cuda:0

print(net(t_gpu))
# tensor([-0.1970,  0.0273], device='cuda:0', grad_fn=<AddBackward0>)

t_cpu = torch.tensor([0.1, 0.2], device='cpu')
print(t_cpu.device)
# cpu

# print(net(t_cpu))
# RuntimeError: Tensor for 'out' is on CPU, Tensor for argument #1 'self' is on CPU, but expected them to be on GPU (while checking arguments for addmm)

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

t = torch.tensor([0.1, 0.2], device=device)

torch.manual_seed(0)
net = torch.nn.Linear(2, 2)
net.to(device)

print(net(t_gpu))
# tensor([-0.1970,  0.0273], device='cuda:0', grad_fn=<AddBackward0>)

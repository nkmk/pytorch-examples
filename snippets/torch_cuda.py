import torch

print(torch.__version__)
# 1.7.1

print(torch.cuda.is_available())
# True

print(torch.cuda.device_count())
# 1

print(torch.cuda.current_device())
# 0

print(torch.cuda.get_device_name())
# GeForce GTX 1080 Ti

print(torch.cuda.get_device_capability())
# (6, 1)

print(torch.cuda.get_device_name(0))
# GeForce GTX 1080 Ti

print(torch.cuda.get_device_name(torch.device('cuda:0')))
# GeForce GTX 1080 Ti

print(torch.cuda.get_device_name('cuda:0'))
# GeForce GTX 1080 Ti

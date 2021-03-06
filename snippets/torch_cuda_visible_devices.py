import os

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import torch

print(torch.cuda.is_available())
# False

print(torch.cuda.device_count())
# 0

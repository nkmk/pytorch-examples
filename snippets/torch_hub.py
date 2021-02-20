import torch

print(torch.__version__)
# 1.7.1

model = torch.hub.load('facebookresearch/pytorch_GAN_zoo:hub', 'DCGAN',
                       pretrained=True, useGPU=False)
# Average network found !
# 
# Using cache found in /Users/mbp/.cache/torch/hub/facebookresearch_pytorch_GAN_zoo_hub

print(torch.hub.list('facebookresearch/pytorch_GAN_zoo:hub'))
# ['DCGAN', 'PGAN']
# 
# Using cache found in /Users/mbp/.cache/torch/hub/facebookresearch_pytorch_GAN_zoo_hub

print(torch.hub.help('facebookresearch/pytorch_GAN_zoo:hub', 'DCGAN'))
# 
#     DCGAN basic model
#     pretrained (bool): load a pretrained model ? In this case load a model
#     trained on fashionGen cloth
#     
# 
# Using cache found in /Users/mbp/.cache/torch/hub/facebookresearch_pytorch_GAN_zoo_hub

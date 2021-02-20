import torch
from torchvision import models, transforms

from PIL import Image

import json

print(torch.__version__)
# 1.7.1

vgg16 = models.vgg16(pretrained=True)

img_org = Image.open('../data/img/src/baboon.jpg')
print(img_org.size)
# (512, 512)

preprocess = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

img = preprocess(img_org)
print(type(img))
# <class 'torch.Tensor'>

print(img.shape)
# torch.Size([3, 224, 224])

img_batch = img[None]
print(img_batch.shape)
# torch.Size([1, 3, 224, 224])

print(torch.unsqueeze(img, 0).shape)
# torch.Size([1, 3, 224, 224])

vgg16.eval()
# VGG(
#   (features): Sequential(
#     (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (1): ReLU(inplace=True)
#     (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (3): ReLU(inplace=True)
#     (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#     (5): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (6): ReLU(inplace=True)
#     (7): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (8): ReLU(inplace=True)
#     (9): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#     (10): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (11): ReLU(inplace=True)
#     (12): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (13): ReLU(inplace=True)
#     (14): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (15): ReLU(inplace=True)
#     (16): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#     (17): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (18): ReLU(inplace=True)
#     (19): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (20): ReLU(inplace=True)
#     (21): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (22): ReLU(inplace=True)
#     (23): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#     (24): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (25): ReLU(inplace=True)
#     (26): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (27): ReLU(inplace=True)
#     (28): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (29): ReLU(inplace=True)
#     (30): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#   )
#   (avgpool): AdaptiveAvgPool2d(output_size=(7, 7))
#   (classifier): Sequential(
#     (0): Linear(in_features=25088, out_features=4096, bias=True)
#     (1): ReLU(inplace=True)
#     (2): Dropout(p=0.5, inplace=False)
#     (3): Linear(in_features=4096, out_features=4096, bias=True)
#     (4): ReLU(inplace=True)
#     (5): Dropout(p=0.5, inplace=False)
#     (6): Linear(in_features=4096, out_features=1000, bias=True)
#   )
# )

result = vgg16(img_batch)
print(type(result))
# <class 'torch.Tensor'>

print(result.shape)
# torch.Size([1, 1000])

idx = torch.argmax(result[0])
print(idx)
# tensor(372)

print(idx.ndim)
# 0

with open('../data/imagenet_class_index.json') as f:
    labels = json.load(f)

print(type(labels))
# <class 'dict'>

print(len(labels))
# 1000

print(labels['0'])
# ['n01440764', 'tench']

print(labels['999'])
# ['n15075141', 'toilet_tissue']

print(labels[str(idx.item())])
# ['n02486410', 'baboon']

probabilities = torch.nn.functional.softmax(result, dim=1)[0]
print(probabilities.shape)
# torch.Size([1000])

print(probabilities.sum())
# tensor(1.0000, grad_fn=<SumBackward0>)

print(probabilities[idx])
# tensor(0.5274, grad_fn=<SelectBackward>)

print(probabilities[idx.item()])
# tensor(0.5274, grad_fn=<SelectBackward>)

_, indices = torch.sort(result[0], descending=True)
print(indices.shape)
# torch.Size([1000])

for idx in indices[:5]:
    print(labels[str(idx.item())][1], ':', probabilities[idx].item())
# baboon : 0.5274456143379211
# guenon : 0.2361937314271927
# patas : 0.08894255757331848
# vulture : 0.05698851868510246
# crane : 0.015375789254903793

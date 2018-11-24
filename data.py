from __future__ import print_function
import zipfile
import os
import torch

import torchvision.transforms as transforms

# once the images are loaded, how do we pre-process them before being passed into the network
# by default, we resize the images to 32 x 32 in size
# and normalize them to mean = 0 and standard-deviation = 1 based on statistics collected from
# the training set
data_transforms = transforms.Compose([
    transforms.CenterCrop(300),
    transforms.RandomAffine(degrees=180, translate=(0.1, 0.1), scale=(0.8, 1.25)),
    transforms.RandomResizedCrop(200, scale=(0.9, 1.1), ratio=(0.67, 1.5)),
    transforms.Scale(120),
    transforms.ColorJitter(0.2, 0.2, 0.2, 0.2),

    transforms.ToTensor(),
    transforms.Normalize((0.3337, 0.3064, 0.3171), (0.2672, 0.2564, 0.2629))
])

val_transforms_crop = transforms.Compose([
    transforms.Scale(180),
    transforms.FiveCrop(120),
    transforms.Lambda(lambda crops: torch.stack([transforms.Normalize((0.3337, 0.3064, 0.3171), (0.2672, 0.2564, 0.2629))(transforms.ToTensor()(crop)) for crop in crops]))
])

val_transforms = transforms.Compose([
    transforms.Scale(180),
    transforms.CenterCrop(120),

    transforms.ToTensor(),
    transforms.Normalize((0.3337, 0.3064, 0.3171), (0.2672, 0.2564, 0.2629))
])

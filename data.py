from __future__ import print_function
import zipfile
import os
import torch

import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

def rotate_crop(img, size):
    img = TF.center_crop(img, size)
    img1 = TF.rotate(img, 90)
    img2 = TF.rotate(img, 180)
    img3 = TF.rotate(img, -90)
    img4 = TF.hflip(img)
    img5 = TF.rotate(img4, 90)
    img6 = TF.rotate(img4, 180)
    img7 = TF.rotate(img4, -90)

    return (img, img1, img2, img3, img4, img5, img6, img7)

class RotateCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img):
        return rotate_crop(img, self.size)

    def __repr__(self):
        return self.__class__.__name__ + '(rotate_crop=true, size={0})'.format(self.size)

# once the images are loaded, how do we pre-process them before being passed into the network
# by default, we resize the images to 32 x 32 in size
# and normalize them to mean = 0 and standard-deviation = 1 based on statistics collected from
# the training set
data_transforms = transforms.Compose([
    transforms.CenterCrop(300),
    transforms.RandomAffine(degrees=90, translate=(0.1, 0.1), scale=(0.8, 1.25)),
    transforms.RandomResizedCrop(200, scale=(0.9, 1.1), ratio=(0.67, 1.5)),
    transforms.RandomHorizontalFlip(0.5),
    transforms.Scale(120),
    transforms.ColorJitter(0.5, 0, 0, 0),

    transforms.ToTensor(),
    transforms.Normalize((0.3337, 0.3064, 0.3171), (0.2672, 0.2564, 0.2629))
])

val_transforms_rotate = transforms.Compose([
    transforms.Scale(180),
    RotateCrop(120),
    transforms.Lambda(lambda crops: torch.stack([transforms.Normalize((0.3337, 0.3064, 0.3171), (0.2672, 0.2564, 0.2629))(transforms.ToTensor()(crop)) for crop in crops]))
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

eps=1e-8

def constraints(x):
    for i in range(x.size()[0]):
        #1
        t = x[i][0].item() + x[i][1].item() + x[i][2].item() + eps
        x[i][0] = x[i][0].item() / t
        x[i][1] = x[i][1].item() / t
        x[i][2] = x[i][2].item() / t

        #2
        t = x[i][3].item() + x[i][4].item() + eps
        x[i][3] = x[i][3].item() / t * x[i][1].item()
        x[i][4] = x[i][4].item() / t * x[i][1].item()

        #7
        t1 = x[i][15].item() + x[i][16].item() + x[i][17].item() + eps
        t2 = x[i][0].item()
        x[i][15] = x[i][15].item() / t1 * t2
        x[i][16] = x[i][16].item() / t1 * t2
        x[i][17] = x[i][17].item() / t1 * t2

        #3
        t1 = x[i][5].item() + x[i][6].item() + eps
        t2 = x[i][4].item()
        x[i][5] = x[i][5].item() / t1 * t2
        x[i][6] = x[i][6].item() / t1 * t2

        #9
        t1 = x[i][25].item() + x[i][26].item() + x[i][27].item() + eps
        t2 = x[i][3].item()
        x[i][25] = x[i][25].item() / t1 * t2
        x[i][26] = x[i][26].item() / t1 * t2
        x[i][27] = x[i][27].item() / t1 * t2

        #4
        t1 = x[i][7].item() + x[i][8].item() + eps
        t2 = x[i][5].item() + x[i][6].item()
        x[i][7] = x[i][7].item() / t1 * t2
        x[i][8] = x[i][8].item() / t1 * t2

        #10
        t1 = x[i][28].item() + x[i][29].item() + x[i][30].item() + eps
        t2 = x[i][3].item()
        x[i][28] = x[i][28].item() / t1 * t2
        x[i][29] = x[i][29].item() / t1 * t2
        x[i][30] = x[i][30].item() / t1 * t2

        #11
        t1 = x[i][31].item() + x[i][32].item() + x[i][33].item() + x[i][34].item() + x[i][35].item() + x[i][36].item() + eps
        t2 = x[i][28].item() + x[i][29].item() + x[i][30].item()
        x[i][31] = x[i][31].item() / t1 * t2
        x[i][32] = x[i][32].item() / t1 * t2
        x[i][33] = x[i][33].item() / t1 * t2
        x[i][34] = x[i][34].item() / t1 * t2
        x[i][35] = x[i][35].item() / t1 * t2
        x[i][36] = x[i][36].item() / t1 * t2

        #5
        t1 = x[i][9].item() + x[i][10].item() + x[i][11].item() + x[i][12].item() + eps
        t2 = x[i][31].item() + x[i][32].item() + x[i][33].item() + x[i][34].item() + x[i][35].item() + x[i][36].item() + x[i][8].item()
        x[i][9] = x[i][9].item() / t1 * t2
        x[i][10] = x[i][10].item() / t1 * t2
        x[i][11] = x[i][11].item() / t1 * t2
        x[i][12] = x[i][12].item() / t1 * t2

        #6
        t1 = x[i][13].item() + x[i][14].item() + eps
        t2 = x[i][9].item() + x[i][10].item() + x[i][11].item() + x[i][12].item() + x[i][15].item() + x[i][16].item() + x[i][17].item() + x[i][25].item() + x[i][26].item() + x[i][27].item()
        x[i][13] = x[i][13].item() / t1 * t2
        x[i][14] = x[i][14].item() / t1 * t2

        #8
        t1 = x[i][18].item() + x[i][19].item() + x[i][20].item() + x[i][21].item() + x[i][22].item() + x[i][23].item() + x[i][24].item() + eps
        t2 = x[i][13].item()
        x[i][18] = x[i][18].item() / t1 * t2
        x[i][19] = x[i][19].item() / t1 * t2
        x[i][20] = x[i][20].item() / t1 * t2
        x[i][21] = x[i][21].item() / t1 * t2
        x[i][22] = x[i][22].item() / t1 * t2
        x[i][23] = x[i][23].item() / t1 * t2
        x[i][24] = x[i][24].item() / t1 * t2

    return x

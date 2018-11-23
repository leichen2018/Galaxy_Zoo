import os
import cv2
import torch
import pickle
import numpy as np
import torch.utils.data as data
from PIL import Image
from random import randint
import math
from torch.utils.data import DataLoader
import torchvision.transforms.functional as f

INPUTSIZE = (224, 224)

class GalaxyZooDataset(data.Dataset):
    dataset_paths = {
        'train_images': '/scratch/yc3390/project/galaxy/data/all/training_files_path.pkl',
        'test_images' :  '/scratch/yc3390/project/galaxy/data/all/test_files_path.pkl',
        'train_probs' :  '/scratch/yc3390/project/galaxy/data/all/training_solutions.pkl',
    }

    def __init__(self, train=True, transform=None):
        self.train = train
        self.transform = transform

        # Prepare dataset
        self.training_set = pickle.load(open(self.dataset_paths['train_images'], 'rb'))
        self.test_set     = pickle.load(open(self.dataset_paths['test_images'], 'rb'))
        self.training_ps  = pickle.load(open(self.dataset_paths['train_probs'], 'rb'))

        self.classes = 37
        self.len_training = len(self.training_set)
        self.len_test     = len(self.test_set)
     
        self.training_keys   = list(self.training_set.keys())
        self.test_keys    = list(self.test_set.keys())

    @staticmethod
    def _list_append(meta, name, x):
        if name in meta:
            meta[name].append(x)
        else:
            meta[name] = [x]

    @staticmethod
    def _meta_stack(meta):
        try:
            meta['image'] = torch.stack(meta['image'])
        except:
            print('Error in stacking metas!')
        meta['prob']      = torch.stack(meta['prob'])
        meta['name']      = torch.stack(meta['name'])

    def collate(self, batch):
        meta = {}
        for item in batch:
            self._list_append(meta, 'image', item['image'])
            self._list_append(meta, 'prob', item['prob'])
            self._list_append(meta, 'name', item['name'])
        self._meta_stack(meta)
        return meta


    def _meta(self, meta, name, image, prob):

        def meta_set(meta, key, val):
            meta[key] = val

        meta_set(meta, 'image', image)
        meta_set(meta, 'name', torch.Tensor([name]))
        meta_set(meta, 'prob', prob)

    def __getitem__(self, index):
        meta = {}
        pic = ''
        name = 0

        if(self.train):
            name = self.training_keys[index]
            pic  = self.training_set[name]
            prob = torch.FloatTensor(self.training_ps[name])

            try:
                image = cv2.imread(pic)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(image)
                image = self.transform(image)

                self._meta(meta, name, image, prob)

            except:
                print('Error in loading image ', pic)
        else:
            name = self.test_keys[index]
            pic  = self.test_set[name]
            prob = torch.Tensor([0]) 

            try:
                image = cv2.imread(pic)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(image)
                image = f.five_crop(image)

                image = torch.stack([f.normalize(f.to_tensor(x)) for x in image], 0)

                self._meta(meta, name, image, prob)

            except:
                print('Error in loading testing image', pic)

        return meta

    def __len__(self):
        if self.train:
            return self.training_set.__len__()
        else:
            return self.test_set.__len__()

if __name__ == '__main__':
    import torchvision.transforms as transforms
    imagenet_mean = (0, 0, 0)
    imagenet_std = (1, 1, 1)
    train_transform = transforms.Compose([
                                    # transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
                                    # transforms.RandomResizedCrop(args.input_size),
                                    transforms.Resize((224, 224)),
                                    # transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    #transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
                                    ])

    train_data = GalaxyZooDataset(train=False, transform=train_transform)
    train_loader = DataLoader(train_data, batch_size=64, shuffle=False,
                                  num_workers=8, pin_memory=True, collate_fn=train_data.collate)
    import cv2
    print(len(train_loader))
    for batch_idx, meta in enumerate(train_loader):
        try:
            print('Nice a there is no error', meta['image'].shape, meta['prob'].shape, batch_idx)
            print(meta['name'])
        except:
            print(meta['name'])
    print('Finished! ')

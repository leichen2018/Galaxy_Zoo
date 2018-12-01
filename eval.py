from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from datetime import datetime as dt
import numpy as np

# Training settings
parser = argparse.ArgumentParser(description='Galaxy ZOO')
parser.add_argument('--name', type=str, default='experiment.csv')
parser.add_argument('--load', type=str)
parser.add_argument('--crop', action='store_true')
parser.add_argument('--rotate', action='store_true')

args = parser.parse_args()
print(args)

### Data Initialization and Loading
from data import data_transforms, val_transforms_crop, val_transforms, constraints, val_transforms_rotate # data.py in the same folder
from galaxy import GalaxyZooDataset
from torch.utils.data import DataLoader

if args.crop:
    transforms = val_transforms_crop
elif args.rotate:
    transforms = val_transforms_rotate
else:
    transforms = val_transforms

val_data = GalaxyZooDataset(train=False, transform=transforms)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False,
                                  num_workers=8, pin_memory=True, collate_fn=val_data.collate)

### Neural Network and Optimizer
# We define neural net in model.py so that it can be reused by the evaluate.py script
#from model_dnn import Net
#from paper_2stn import Net
#from paper_tund import Net
from paper_groupy import Net
model = Net()
device = torch.device('cuda:0')

if args.load:
    try:    
        model.load_state_dict(torch.load(args.load))
        print("Load sucessfully !", args.load)
    except:
        print("Training from scratch!")

model.to(device)
if args.crop:
    output_file = open('./results/' + args.name + '_crop', "w")
elif args.rotate:
    output_file = open('./results/' + args.name + '_rotate', "w")
else:
    output_file = open('./results/' + args.name + '_nocrop', "w")


head = 'GalaxyID,Class1.1,Class1.2,Class1.3,Class2.1,Class2.2,Class3.1,Class3.2,Class4.1,Class4.2,Class5.1,Class5.2,Class5.3,Class5.4,Class6.1,Class6.2,Class7.1,Class7.2,Class7.3,Class8.1,Class8.2,Class8.3,Class8.4,Class8.5,Class8.6,Class8.7,Class9.1,Class9.2,Class9.3,Class10.1,Class10.2,Class10.3,Class11.1,Class11.2,Class11.3,Class11.4,Class11.5,Class11.6\n'
output_file.write(head)


def validation():
    model.eval()
    for meta in val_loader:
        if args.crop:
            data = meta['image'].view(-1, 3, 120, 120)
            data = data.to(device)
        else:
            data = meta['image'].to(device)
        names = meta['name']
        data = Variable(data, volatile=True)
        output = torch.clamp(model(data), 0, 1)
        
        if args.crop:
            output = output.view(len(names), 5, -1).max(1)
            output = output[0]
        elif args.rotate:
            output = output.view(len(names), 4, -1).mean(1)

        #output = constraints(output)

        for i in range(len(names)):
            name = names[i]
            strs = "{}".format(int(name))
            for j in range(37):
                strs = strs + ',{}'.format(float(output[i][j]))
            output_file.write(strs + '\n')

    print(dt.now(), 'Done. ')
##


print(dt.now(),'Start.') 
validation()
output_file.close()

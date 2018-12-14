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
import os
import math

# Training settings
parser = argparse.ArgumentParser(description='PyTorch GTSRB example')
parser.add_argument('--name', type=str, default='experiment', metavar='NM',
                    help="name of the training")
parser.add_argument('--load', type=str,
                    help="load previous model to finetune")
parser.add_argument('--ftn', type=str,
                    help="fine tune number")
parser.add_argument('--data', type=str, default='data', metavar='D',
                    help="folder where data is located. train_data.zip and test_data.zip need to be found in the folder")
parser.add_argument('--no_dp', action='store_true', default=False,
                    help="if there is no dropout")
parser.add_argument('--batch_size', type=int, default=64, metavar='B',
                    help='input batch size for training (default: 64)')
parser.add_argument('--step', type=int, default=10, metavar='S', 
                    help='lr decay step (default: 5)')
parser.add_argument('--epochs', type=int, default=50, metavar='N',
                    help='number of epochs to train (default: 30)')
parser.add_argument('--p', type=float, default=0.5)
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--weight_decay', '-wd', type=float, default=1e-4, metavar='WD',
                    help='Weight decay (default: 1e-4)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--rc', action='store_true')
parser.add_argument('--optimized', action='store_true')
args = parser.parse_args()
print(args)

torch.manual_seed(args.seed)

### Data Initialization and Loading
from data import data_transforms, val_transforms, data_transforms_rc # data.py in the same folder
from galaxy import GalaxyZooDataset
from torch.utils.data import DataLoader

if args.rc:
    transform = data_transforms_rc
else:
    transform = data_transforms

train_data = GalaxyZooDataset(train=True, transform=transform)
train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                                  num_workers=8, pin_memory=True, collate_fn=train_data.collate)

### Neural Network and Optimizer
# We define neural net in model.py so that it can be reused by the evaluate.py script
#from model_dnn import Net
#from paper_2stn import Net
#from paper_tund import Net
from paper_groupy import Net
model = Net(args.no_dp, p = args.p, optimized = args.optimized)
device = torch.device('cuda:0')

if args.load:
    try:    
        model.load_state_dict(torch.load(args.load))
        print("Load sucessfully !", args.load)
    except:
        print("Training from scratch!")

model.to(device)
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay = args.weight_decay)
if args.load:
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30,60], gamma=0.2)
elif args.rc:
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15, 25, 35], gamma=0.1)
else:
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150, 200], gamma=0.1)

least_mse = np.inf

def train(epoch):
    model.train()
    correct = 0
    loss_total = 0
    loss_step  = 0
    for batch_idx, meta in enumerate(train_loader):

        if args.rc:
            data, target = meta['image'].view(-1, 3, 120, 120).to(device), meta['prob'].to(device)
        else:
            data, target = meta['image'].to(device), meta['prob'].to(device)

        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()

        if args.rc:
            a = torch.cuda.FloatTensor(16, target.size()[0], 37).fill_(0)
            target = a + target
            target = torch.transpose(target, 0, 1)
            target = target.reshape(-1, 37)
        
        output = model(data)

        loss = F.mse_loss(output, target)

        loss.backward()
        optimizer.step()
        loss_total += float(loss.data[0])
        loss_step  += float(loss.data[0])

        if batch_idx % args.log_interval == 0 and batch_idx != 0:
            print(dt.now(), 'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} '.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss_step / args.log_interval ))
            loss_step = 0

    print("\nTraining MSE loss: ", math.sqrt(loss_total * 1.0 / len(train_loader)))
    return math.sqrt(loss_total * 1.0 / len(train_loader))       

if args.load:
    os.mkdir("models/" + args.name + '_ft' + args.ftn)

for epoch in range(1, args.epochs + 1):
    loss = train(epoch)
    scheduler.step()
    if args.load:
        model_file = "models/" + args.name + '_ft' + args.ftn  +'/model_' + str(epoch) +'_{:.9f}'.format(loss) + '.pth'
    else:
        model_file = "models/" + args.name +'/model_' + str(epoch) +'_{:.9f}'.format(loss) + '.pth'
    if loss < least_mse :
        least_mse = loss
        torch.save(model.state_dict(), model_file)
        print('\nSaved model to ' + model_file )

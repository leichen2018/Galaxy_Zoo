from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from datetime import datetime as dt
from tqdm import tqdm
import os

from IPython import embed
# Training settings
parser = argparse.ArgumentParser(description='PyTorch GTSRB example')
parser.add_argument('--name', type=str, default='experiment', metavar='NM',
                    help="name of the training")
parser.add_argument('--load', type=str,
                    help="load previous model to finetune")
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
args = parser.parse_args()
print(args)

torch.manual_seed(args.seed)

### Data Initialization and Loading
from data import initialize_data, data_transforms, val_transforms # data.py in the same folder
initialize_data(args.data) # extracts the zip files, makes a validation set

train_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(args.data + '/train_images',
                         transform=data_transforms),
    batch_size=args.batch_size, shuffle=True, num_workers=4)
val_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(args.data + '/val_images',
                         transform=val_transforms),
    batch_size=1, shuffle=False, num_workers=8)

### Neural Network and Optimizer
# We define neural net in model.py so that it can be reused by the evaluate.py script
#from model_dnn import Net
from paper_stn import Net
model = Net(args.no_dp)
device = torch.device('cuda:0')

if args.load:
    try:    
        model.load_state_dict(torch.load(args.load))
        print("Load sucessfully !", args.load)
    except:
        raise RuntimeError('No module loaded!')
        print("Training from scratch!")

model.to(device)
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay = args.weight_decay)
scheduler = optim.lr_scheduler.StepLR(optimizer, args.step)
best_accu = 0
wrongs = []
img_dir = './data/test_images/'
imgs = os.listdir('./data/test_images')
wrongs_dir = './wrongs/' + args.name + '/'
os.system('mkdir -p '+ wrongs_dir)


def validation():
    model.eval()
    validation_loss = 0
    correct = 0
    count = 0
    for idx, (data, target) in tqdm( enumerate(val_loader)):
        data, target = data.to(device), target.to(device)
        data, target = Variable(data), Variable(target)
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        validation_loss += F.nll_loss(output, target, size_average=False).data[0] # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        k = pred.eq(target.data.view_as(pred)).cpu().sum()

        if(k == 0):
            count = count + 1
            wrongs.append(imgs[idx])
            print(idx, " wrong at ", imgs[idx])
            os.system('cp {} {}'.format(img_dir + imgs[idx], wrongs_dir + imgs[idx]))
            print('Done cp {} {}'.format(img_dir + imgs[idx], wrongs_dir + imgs[idx]))
    print("wrongs ", count)



    validation_loss /= len(val_loader.dataset)
    print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        validation_loss, correct, len(val_loader.dataset),
        100. * int(correct) / len(val_loader.dataset)))

    return 100. * int(correct) / len(val_loader.dataset)


accu = validation()

print("Accuracy: ", accu)


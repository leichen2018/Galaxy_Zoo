from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from datetime import datetime as dt

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
    batch_size=args.batch_size, shuffle=False, num_workers=8)

### Neural Network and Optimizer
# We define neural net in model.py so that it can be reused by the evaluate.py script
#from model_dnn import Net
from paper_conv0_stn import Net
model = Net(args.no_dp)
device = torch.device('cuda:0')

if args.load:
    try:    
        model.load_state_dict(torch.load(args.load))
        print("Load sucessfully !", args.load)
    except:
        print("Training from scratch!")

model.to(device)
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay = args.weight_decay)
scheduler = optim.lr_scheduler.StepLR(optimizer, args.step)
best_accu = 0

def train(epoch):
    model.train()
    correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

        #scheduler.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, Accuracy: {}/{} '.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0], correct, args.log_interval * len(data)),'({:.2f}%)'.format( float(100.00) * int(correct) * 1.0 / float(args.log_interval * len(data) )) )
            correct = 0;

def validation():
    model.eval()
    validation_loss = 0
    correct = 0
    for data, target in val_loader:
        data, target = data.to(device), target.to(device)
        data, target = Variable(data), Variable(target)
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        validation_loss += F.nll_loss(output, target, size_average=False).data[0] # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    validation_loss /= len(val_loader.dataset)
    print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        validation_loss, correct, len(val_loader.dataset),
        100. * int(correct) / len(val_loader.dataset)))

    return 100. * int(correct) / len(val_loader.dataset)

for epoch in range(1, args.epochs + 1):
    train(epoch)
    accu = validation()
    scheduler.step()
    model_file = "models/" + args.name +'/model_' + str(epoch) +'_{:.2f}'.format(accu) + '.pth'
    if accu > best_accu and accu > 98:
        best_accu = accu
        torch.save(model.state_dict(), model_file)
        print('\nSaved model to ' + model_file + '. You can run `python evaluate.py ' + model_file + '` to generate the Kaggle formatted csv file')

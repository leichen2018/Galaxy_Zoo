import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import math

nclasses = 43 # GTSRB as 43 classes

class Net(nn.Module):
    def __init__(self, no_dp=False):
        super(Net, self).__init__()
        self.no_dp = no_dp
        self.conv1 = nn.Conv2d(3, 100, kernel_size=7)
        self.bn1 = nn.BatchNorm2d(100)
        self.bn0 = nn.BatchNorm2d(3)
        self.conv2 = nn.Conv2d(100, 150, kernel_size=4)
        self.conv3 = nn.Conv2d(150, 250, kernel_size=4)
        self.bn2 = nn.BatchNorm2d(250)
        if not no_dp:
            self.conv2_drop = nn.Dropout2d()
            self.conv3_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(2250, 300)
        self.fc2 = nn.Linear(300, nclasses)

        # Initilize the parameters
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            '''
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            '''

        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 3 * 3, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    # Spatial transformer network forward function
    def stn(self, x, x1):
        xs = self.localization(x1)
        xs = xs.view(-1, 10 * 3 * 3)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        return x

    def forward(self, x):
        # STN 1
        x1 = F.upsample(x, size=(28, 28), mode='bilinear')
        x = self.stn(x, x1)

        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = self.bn1(x)

        if self.no_dp:
            x = F.relu(F.max_pool2d(self.conv2(x), 2))
            x = F.relu(F.max_pool2d(self.conv3(x), 2))
        else:
            x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
            x = F.relu(F.max_pool2d(self.conv3_drop(self.conv3(x)), 2))

        x = self.bn2(x)
        x = x.view(-1, 2250)
        x = F.relu(self.fc1(x))
        if not self.no_dp:
            x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)

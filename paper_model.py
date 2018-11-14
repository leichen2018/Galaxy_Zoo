import torch
import torch.nn as nn
import torch.nn.functional as F
import math

nclasses = 43 # GTSRB as 43 classes

class Net(nn.Module):
    def __init__(self, no_dp=False):
        super(Net, self).__init__()
        self.no_dp = no_dp
        self.conv1 = nn.Conv2d(3, 100, kernel_size=7)
        self.bn1 = nn.BatchNorm2d(100)

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

    def forward(self, x):
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
            x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)

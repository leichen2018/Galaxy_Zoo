import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import torch
from custom import OptimisedDivGalaxyOutputLayer 

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

def partial_nll(scores, target):
    '''First 3 answers sums up to 1, use a negative log likelihood loss to correct it further.
    '''
    epi = 1e-12
    probs = scores[:, :3] 
    #probs = scores[:, :3] / (socres[:, :3].sum(dim=1) + epi)
    g_t = target[:, :3]
    return F.nll_loss(probs, g_t) 

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, mid_layer=500, num_classes=37, dp=0.5, lock_bn=False, sigmoid=False, optimized=True):
        self.inplanes = 64
        self.optimized = optimized
        self.dp = dp
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc1 = nn.Linear(512 * block.expansion, mid_layer)
        self.fc2 = nn.Linear(mid_layer, num_classes)
        self.score = nn.Sigmoid()
        self.sigmoid = sigmoid
        self.optimized_output = OptimisedDivGalaxyOutputLayer() 

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                if(lock_bn):
                    m.weight.requires_grad= False
                    m.bias.requires_grad  = False

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = F.dropout(x, p=self.dp, training=self.training)
        x = self.relu(self.fc1(x))
        ### Double Dropout
        x = F.dropout(x, p=self.dp, training=self.training)
        x = self.fc2(x)

        if self.optimized:
            x = self.relu(x)
            x = self.optimized_output.predictions(x) 
        elif self.sigmoid:
            x = self.score(x)
            x = self.optimized_output.predictions(x) 
        else:
            x = torch.clamp(x, 0, 1)
        return x


def resnet18(pretrained=False, dp=0.0, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], dp=dp,  **kwargs)
    if pretrained:
        try:
            model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
        except:
            print('Last dimension size mismatch!')
            
    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], dp=0.2, **kwargs)
    if pretrained:
        try:
            model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
        except:
            print('Last dimension size mismatch!')

    return model


def resnet50(pretrained=False,dp=0.2, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3],dp=dp, **kwargs)
    state_dict = model.state_dict()
    if pretrained:
        try:
            model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
        except:
            print('Last dimension size mismatch!')

            '''
            pretrained_file = '~/.torch/models/' + model_urls['resnet50'].strip()[-1] 
            raw_dict = torch.load(pretrained_file) 
            for k,v in raw_dict.items():
                if 'fc' in k:
                    continue;
                if isinstance(v, torch.nn.parameter.Parameter):
                    v = v.data
                    state_dict[k].copy_(v)
            '''
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        try:
            model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
        except:
            print('Last dimension size mismatch!')

    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        try:
            model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
        except:
            print('Last dimension size mismatch!')

    return model

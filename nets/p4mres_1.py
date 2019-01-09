import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import torch
from custom import OptimisedDivGalaxyOutputLayer 
from groupy.gconv.pytorch_gconv import P4MConvZ2, P4MConvP4M
from groupy.gconv.pytorch_gconv.splitgconv2d import SplitGConv2D
from groupy.gconv.pytorch_gconv.pooling import plane_group_spatial_max_pooling

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

class shaper(nn.Module):
    def __init__(self, channels=8, to_groupy=False):
        super(shaper, self).__init__()
        self.to_groupy = to_groupy
        self.channels = channels

    def forward(self, x):
        xs = x.size()
        if self.to_groupy:
            x = x.view(xs[0], xs[1] // self.channels, self.channels, xs[2], xs[3])
        else:
            x = x.view(xs[0], xs[1] * xs[2], xs[3], xs[4])
        return x

class groupy_bn(nn.Module):
    def __init__(self, batch_channels):
        super(groupy_bn, self).__init__()
        self.bn = nn.BatchNorm3d(batch_channels)

    def forward(self, x):
        x = self.bn(x)
        return x

def plane_group_spatial_avg_pooling(x, ksize, stride=None, pad=0):
    xs = x.size()
    x = x.view(xs[0], xs[1] * xs[2], xs[3], xs[4])
    x = F.avg_pool2d(input=x, kernel_size=xs[3], stride=None, padding=pad)
    x = x.view(xs[0], xs[1], xs[2], x.size()[2], x.size()[3])
    return x

class groupy_pooling(nn.Module):
    def __init__(self, ksize=0, stride=None, pad=0, avg=False):
        super(groupy_pooling, self).__init__()
        if avg:
            self.pooling = plane_group_spatial_avg_pooling
        else:
            self.pooling = plane_group_spatial_max_pooling
        self.ksize = ksize
        self.stride = stride
        self.pad = pad

    def forward(self, x):
        x = self.pooling(x, self.ksize, self.stride, self.pad)
        return x


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, channels=8, downsample=None):
        super(BasicBlock, self).__init__()
        self.to_groupy = shaper(to_groupy=True)
        self.to_normal = shaper(to_groupy=False)
        self.channels = channels
        self.conv1 = P4MConvP4M(inplanes, planes, kernel_size=3, stride=stride,padding=1)
        #self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = groupy_bn(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = P4MConvP4M(planes, planes, kernel_size=3, stride=1,padding=1)
        #self.conv2 = conv3x3(planes, planes)
        #self.bn2 = nn.BatchNorm2d(planes)
        self.bn2 = groupy_bn(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        #print("res ", residual.shape)
        #x = self.to_groupy(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            #print('before downsample ', residual.shape)
            residual = self.downsample(residual)

        #print('out ',out.shape, " res ", residual.shape)
        out += residual
        out = self.relu(out)

        return out



class ResNet(nn.Module):

    def __init__(self, block, layers, mid_layer=500, num_classes=37, dp=0.5, lock_bn=False, sigmoid=False, optimized=True):
        self.inplanes = 21
        self.optimized = optimized
        self.dp = dp
        super(ResNet, self).__init__()
        #self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
        #                       bias=False)
        self.conv1 = P4MConvZ2(3, 21, kernel_size=7, stride=2, padding=3, bias=False)
        self.to_normal = shaper(to_groupy=False)
        self.bn1 = nn.BatchNorm3d(21)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = groupy_pooling(ksize=3, stride=2, pad=1)

        self.layer1 = self._make_layer(block, 21, layers[0])
        self.layer2 = self._make_layer(block, 42, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 85, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 170, layers[3], stride=2)
        self.avgpool = groupy_pooling(avg=True)
        self.fc1 = nn.Linear(170 *8 * block.expansion, mid_layer)
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
                P4MConvP4M(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(planes * block.expansion)
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        print(x.size)

        x = self.bn1(x)
        x = self.relu(x)
        print(x.size)

        x = self.maxpool(x)
        print(x.size)

        x = self.layer1(x)
        print(x.size)
        x = self.layer2(x)
        print(x.size)
        x = self.layer3(x)
        print(x.size)
        x = self.layer4(x)
        print(x.size)

        x = self.avgpool(x)
        print(x.size)

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

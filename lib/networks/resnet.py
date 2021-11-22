'''ResNet in PyTorch.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import ResNet, BasicBlock, Bottleneck, load_state_dict_from_url, model_urls

__all__ = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'resnet26']

def _resnet(arch, block, layers, pretrained, progress, model_path=None, **kwargs):
    useful_key =  ['num_classes', 'zero_init_residual', 'groups', 'width_per_group',
                       'replace_stride_with_dilation', 'norm_layer']
    irrelevant_key = [key for key in kwargs if key not in useful_key]
    for key in irrelevant_key:
        del kwargs[key]

    if pretrained:
        num_classes = kwargs['num_classes']
        kwargs['num_classes'] = 1000


        model = ResNet(block, layers, **kwargs)
        if model_path is None:
            state_dict = load_state_dict_from_url(model_urls[arch],
                                                  progress=progress)
            model.load_state_dict(state_dict)
        else:
            print("Loading model from {}".format(model_path))
            checkpoint = torch.load(model_path, map_location='cpu')
            model.load_state_dict(checkpoint['state_dict'])

        if num_classes != 1000:
            model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        model = ResNet(block, layers, **kwargs)

    return model

def resnet18(pretrained=False, progress=True, **kwargs):
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)

def resnet34(pretrained=False, progress=True, **kwargs):
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)

def resnet50(pretrained=False, progress=True, **kwargs):
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)

def resnet101(pretrained=False, progress=True, **kwargs):
    return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained, progress,
                   **kwargs)

def resnet152(pretrained=False, progress=True, **kwargs):
    return _resnet('resnet152', Bottleneck, [3, 8, 36, 3], pretrained, progress,
                   **kwargs)

def resnet26(pretrained=False, progress=True, **kwargs):
    return ResNetCifar(depth=26)


# Based on the ResNet implementation in torchvision
# https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py

import math
import torch
from torch import nn
from torchvision.models.resnet import conv3x3
from torch.nn import functional as F
from torch.autograd import Variable

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# class BasicBlock(nn.Module):
#     def __init__(self, inplanes, planes, norm_layer, stride=1, downsample=None):
#         super(BasicBlock, self).__init__()
#         self.downsample = downsample
#         self.stride = stride
        
#         self.bn1 = norm_layer(inplanes)
#         self.relu1 = nn.ReLU(inplace=True)
#         self.conv1 = conv3x3(inplanes, planes, stride)
        
#         self.bn2 = norm_layer(planes)
#         self.relu2 = nn.ReLU(inplace=True)
#         self.conv2 = conv3x3(planes, planes)

#     def forward(self, x):
#         residual = x 
#         residual = self.bn1(residual)
#         residual = self.relu1(residual)
#         residual = self.conv1(residual)

#         residual = self.bn2(residual)
#         residual = self.relu2(residual)
#         residual = self.conv2(residual)

#         if self.downsample is not None:
#             x = self.downsample(x)
#         return x + residual

# class Downsample(nn.Module):
#     def __init__(self, nIn, nOut, stride):
#         super(Downsample, self).__init__()
#         self.avg = nn.AvgPool2d(stride)
#         assert nOut % nIn == 0
#         self.expand_ratio = nOut // nIn

#     def forward(self, x):
#         x = self.avg(x)
#         return torch.cat([x] + [x.mul(0)] * (self.expand_ratio - 1), 1)

# class ResNetCifar(nn.Module):
#     def __init__(self, depth, width=1, classes=10, channels=3, \
#         norm_layer=nn.BatchNorm2d, b=1):
#         assert (depth - 2) % 6 == 0         # depth is 6N+2
#         self.N = (depth - 2) // 6
#         super(ResNetCifar, self).__init__()

#         # Following the Wide ResNet convention, we fix the very first convolution
#         self.conv1 = nn.Conv2d(channels, 16, kernel_size=3, stride=1, padding=1, bias=False)
#         self.inplanes = 16
#         self.layer1 = self._make_layer(norm_layer, 16 * width)
#         self.layer2 = self._make_layer(norm_layer, 32 * width, stride=2)
#         self.layer3 = self._make_layer(norm_layer, 64 * width, stride=2)
#         self.bn = norm_layer(64 * width)
#         self.relu = nn.ReLU(inplace=True)
#         self.avgpool = nn.AvgPool2d(8)
#         self.fc = nn.Linear(64 * width, classes)

#         # Initialization
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#                 m.weight.data.normal_(0, math.sqrt(2. / n))
                
#     def _make_layer(self, norm_layer, planes, stride=1):
#         downsample = None
#         if stride != 1 or self.inplanes != planes:
#             downsample = Downsample(self.inplanes, planes, stride)
#         layers = [BasicBlock(self.inplanes, planes, norm_layer, stride, downsample)]
#         self.inplanes = planes
#         for i in range(self.N - 1):
#             layers.append(BasicBlock(self.inplanes, planes, norm_layer))
#         return nn.Sequential(*layers)

#     def forward(self, x, return_feat=False, b=[1,1]):
#         x = self.conv1(x)
#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.bn(x)
#         x = self.relu(x)
#         x = self.avgpool(x)
#         x = x.view(x.size(0), -1)
        
#         if return_feat:
#             return x, self.fc(x)
#         else:
#             return self.fc(x)
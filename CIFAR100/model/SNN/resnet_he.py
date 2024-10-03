import torch
import torch.nn as nn
from spikingjelly.clock_driven.neuron import MultiStepParametricLIFNode
from spikingjelly.clock_driven import layer
from typing import Type, Any, Callable, Union, List, Optional
from torch import Tensor


def LDCB3x3(in_channels: int, out_channels: int, stride: int = 1, groups: int = 1, padding: int = 1) -> nn.Sequential:
    return nn.Sequential(
        MultiStepParametricLIFNode(init_tau=2.0, detach_reset=True),
        layer.SeqToANNContainer(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride,
                                          groups=groups, padding=padding, bias=False),
                                nn.BatchNorm2d(out_channels)
                                )
    )


def LCB1x1(in_channels: int, out_channels: int, stride: int = 1, groups: int = 1) -> nn.Sequential:
    return nn.Sequential(
        MultiStepParametricLIFNode(init_tau=2.0, detach_reset=True),
        layer.SeqToANNContainer(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride,
                                          groups=groups, bias=False),
                                nn.BatchNorm2d(out_channels)
                                )
    )


def LCB3x3(in_channels: int, out_channels: int, stride: int = 1, groups: int = 1, padding: int = 1) -> nn.Sequential:
    return nn.Sequential(
        MultiStepParametricLIFNode(init_tau=2.0, detach_reset=True),
        layer.SeqToANNContainer(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=padding,
                                          groups=groups, bias=False),
                                nn.BatchNorm2d(out_channels)
                                )
    )


class SNNBasicBlock(nn.Module):
    expansion = 1

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 downsample=None,
                 radix=2,
                 cardinality=1,
                 norm_layer=None,):
        super(SNNBasicBlock, self).__init__()

        self.radix = radix
        self.downsample = downsample

        self.conv1_s = LDCB3x3(inplanes, planes, stride, groups=inplanes)
        self.conv2_s = LCB3x3((inplanes + planes) // radix, planes)
        if downsample is not None:
            self.maxpool = layer.SeqToANNContainer(nn.MaxPool2d(kernel_size=2, stride=2), )

    def forward(self, x):
        res = x
        out = self.conv1_s(x)
        T, B, C, H, W = out.shape
        out = out.view(T, B, C // self.radix, self.radix, H, W)
        if self.downsample is not None:
            out_gc = self.maxpool(x)
        else:
            out_gc = x

        T_, B_, C_, H_, W_ = out_gc.shape
        out_gc = out_gc.view(T_, B_, C_ // self.radix, self.radix, H_, W_)
        out = torch.cat([out, out_gc], dim=2)
        out = out.sum(dim=3)
        out = self.conv2_s(out)

        if self.downsample is not None:
            res = self.downsample(x)
        out += res
        return out



class SNNConcatBlock(nn.Module):
    expansion: int = 1

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 downsample=None,
                 radix=2,
                 cardinality=1,
                 norm_layer=None,):
        super(SNNConcatBlock, self).__init__()

        self.radix = radix
        self.downsample = downsample

        self.conv1_s = LDCB3x3(inplanes, planes, stride, groups=inplanes)
        self.conv2_s = LCB3x3((inplanes + planes) // radix, planes)
        if downsample is not None:
            self.maxpool = layer.SeqToANNContainer(nn.MaxPool2d(kernel_size=2, stride=2), )

    def forward(self, x):
        res = x
        out = self.conv1_s(x)
        T, B, C, H, W = out.shape
        out = out.view(T, B, C // self.radix, self.radix, H, W)
        if self.downsample is not None:
            out_gc = self.maxpool(x)
        else:
            out_gc = x

        T_, B_, C_, H_, W_ = out_gc.shape
        out_gc = out_gc.view(T_, B_, C_ // self.radix, self.radix, H_, W_)
        out = torch.cat([out, out_gc], dim=2)
        out = out.sum(dim=3)
        out = self.conv2_s(out)

        if self.downsample is not None:
            res = self.downsample(x)
            res = torch.cat((res, x), dim=2)
            res = self.maxpool(res)
        out += res
        return out


class ResNet(nn.Module):
    def __init__(self,
                 name,
                 layers,
                 num_cls,
                 channel=64,
                 radix=2,
                 groups=1):

        super(ResNet, self).__init__()
        self.name = name
        self.radix = radix
        self.inplanes = channel
        self.groups = groups
        self.conv1_s = layer.SeqToANNContainer(nn.Conv2d(3, channel, kernel_size=3, stride=1, padding=1, bias=False),
                                               nn.BatchNorm2d(channel))
        self.layer1 = self._make_layer(SNNBasicBlock, SNNConcatBlock, channel, layers[0], stride=1, name='layer1')
        self.layer2 = self._make_layer(SNNBasicBlock, SNNConcatBlock, channel * 2, layers[1], stride=2, name='layer2')
        self.layer3 = self._make_layer(SNNBasicBlock, SNNConcatBlock, channel * 4, layers[2], stride=2, name='layer3')
        self.layer4 = self._make_layer(SNNBasicBlock, SNNConcatBlock, channel * 8, layers[3], stride=2, name='layer4')

        self.adap_avgpool = layer.SeqToANNContainer(nn.AdaptiveAvgPool2d((1, 1)))
        self.line = nn.Linear(channel * 8, num_cls, bias=True)

    def _make_layer(self,
                    block_Basic: Type[Union[SNNBasicBlock]],
                    block_Concat: Type[Union[SNNConcatBlock]],
                    planes: int,
                    blocks: int,
                    stride: int = 1,
                    name: str = None
                    ) -> nn.Sequential:

        downsample = None
        if stride != 1 or self.inplanes != planes * block_Basic.expansion:
            if name == 'layer1':
                downsample = nn.Sequential(
                    layer.SeqToANNContainer(nn.MaxPool2d(kernel_size=2, stride=2), ),
                    LCB1x1(self.inplanes, planes * block_Basic.expansion)
                )
            else:
                downsample = nn.Sequential(
                    LCB1x1(self.inplanes, planes * block_Concat.expansion // 2)
                )

        layers = []
        if name == 'layer1':
            layers.append(block_Basic(inplanes=self.inplanes,
                                planes=planes,
                                stride=stride,
                                downsample=downsample,
                                radix=self.radix,
                                cardinality=self.groups)
                          )
        else:
            layers.append(block_Concat(inplanes=self.inplanes,
                                 planes=planes,
                                 stride=stride,
                                 downsample=downsample,
                                 radix=self.radix,
                                 cardinality=self.groups,)
                          )

        self.inplanes = planes * block_Basic.expansion
        for _ in range(1, blocks):
            layers.append(block_Basic(self.inplanes, planes,
                                radix=self.radix,
                                cardinality=self.groups)
                          )

        return nn.Sequential(*layers)


    def _forward_impl(self, x: Tensor):
        x = self.conv1_s(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.adap_avgpool(x).flatten(2)
        # x = x.flatten(2)
        x = self.line(x.mean(0))

        return x

    def forward(self, x: Tensor):
        return self._forward_impl(x)


def _resnet(arch: str, layers: List[int], num_cls: int, channel: int) -> ResNet:
    return ResNet(arch, layers, num_cls, channel, radix=4)


def resnet10(num_cls: int, channel: int) -> ResNet:
    return _resnet('resnet10', [1, 1, 1, 1], num_cls, channel)


def resnet18(num_cls: int, channel: int) -> ResNet:
    return _resnet('resnet18', [2, 2, 2, 2], num_cls, channel)


def resnet34(num_cls: int, channel: int) -> ResNet:
    return _resnet('resnet34', [3, 4, 6, 3], num_cls, channel)


def resnet68(num_cls: int, channel: int) -> ResNet:
    return _resnet('resnet68', [3, 4, 23, 3], num_cls, channel)


def resnet104(num_cls: int, channel: int) -> ResNet:
    return _resnet('resnet104', [3, 8, 32, 8], num_cls, channel)


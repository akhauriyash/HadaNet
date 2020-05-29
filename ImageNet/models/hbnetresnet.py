import torch
import os
import time, math, sys
import numpy as np
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
from torch.autograd.function import Function, once_differentiable
from functools import reduce
from torch.nn.modules.module import _addindent
from torch.nn._functions.thnn import _all_functions
from torch.autograd import gradcheck, Variable
from torch._thnn import type2backend
from torch.nn.modules.utils import _pair

# For proving angle preservation property
def angle(a, b):
    cos = nn.CosineSimilarity()
    return math.degrees(math.acos(cos(a.view(1, -1), b.view(1, -1))))


def binFunc(input, abs_bin=False, signed_bin=False, binmat_mul=False):
    binAgg = 2
    shape = input.size()
    # restore = 0
    if len(shape) == 4:
        if shape[-1] > binAgg:
            # restore   = input.size()
            if abs_bin == True:
                listmat = list(torch.split(torch.abs(input), binAgg, dim=-1))
            elif binmat_mul == True:
                listmat = list(torch.split(torch.abs(input), binAgg, dim=-1))
                binmat = input.sign()
            elif signed_bin == True:
                listmat = list(torch.split(input, binAgg, dim=-1))
            rz = listmat[-1].size(-1)
            residualmat = torch.mean(listmat[-1], dim=-1, keepdim=True).repeat(
                1, 1, 1, rz
            )
            listmat = torch.stack(listmat[:-1])
            rz = listmat[0].size(-1)
            listmat = torch.mean(listmat, dim=-1, keepdim=True).repeat(1, 1, 1, 1, rz)
            listmat = torch.cat(list(listmat), dim=-1)
            output = torch.cat((listmat, residualmat), dim=-1)
            if binmat_mul == True:
                output = output.mul(binmat)
        else:
            if abs_bin == True:
                listmat = list(input)
            elif binmat_mul == True:
                listmat = list(input)
                binmat = input.sign()
            elif signed_bin == True:
                listmat = list(input)
            listmat = torch.stack(listmat)
            listmat = torch.mean(listmat, dim=-1, keepdim=True).repeat(
                1, 1, 1, listmat[-1].size(-1)
            )
            output = listmat
            if binmat_mul == True:
                output = output.mul(binmat)
        return output
    shape = input.size()
    if len(shape) == 2:
        input = input.unsqueeze(1)
        shape = input.size()
        if shape[-1] > binAgg:
            if abs_bin == True:
                listmat = list(torch.split(torch.abs(input), binAgg, dim=-1))
            elif binmat_mul == True:
                listmat = list(torch.split(torch.abs(input), binAgg, dim=-1))
                binmat = input.sign()
            elif signed_bin == True:
                listmat = list(torch.split(input, binAgg, dim=-1))
            residualmat = torch.mean(listmat[-1], dim=-1, keepdim=True).repeat(
                1, 1, listmat[-1].size(-1)
            )
            listmat = torch.stack(listmat[:-1])
            listmat = torch.mean(listmat, dim=-1, keepdim=True).repeat(
                1, 1, 1, listmat[0].size(-1)
            )
            listmat = torch.cat(list(listmat), dim=-1)
            output = torch.cat((listmat, residualmat), dim=-1)
            if binmat_mul == True:
                output = output.mul(binmat)
        else:
            if abs_bin == True:
                listmat = list(input)
            elif binmat_mul == True:
                listmat = list(input)
                binmat = input.sign()
            elif signed_bin == True:
                listmat = list(input)
            listmat = torch.stack(listmat)
            listmat = torch.mean(listmat, dim=-1, keepdim=True).repeat(
                1, 1, listmat[-1].size(-1)
            )
            output = listmat
            if binmat_mul == True:
                output = output.mul(binmat)
        output = torch.squeeze(output)
        # if(restore!=0):
        #     output = output.reshape(restore)
        return output


class BinActive(Function):
    @staticmethod
    def forward(ctx, input):
        output = binFunc(input, abs_bin=True)
        ctx.save_for_backward(input, Variable(output))
        output = output.mul(input.sign())
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, output = ctx.saved_tensors
        # output             =    binFunc(input, abs_bin=True)
        binAgg = 4
        grad_input = grad_output.clone()
        g_out = grad_output.clone()
        s = g_out.size()
        grad_input[input.le(-1.0)] = 0
        grad_input[input.ge(1.0)] = 0
        m = output.abs().mul(grad_input)
        m_add = grad_output.mul(input.sign())
        m_add = binFunc(m_add, signed_bin=True).mul(input.sign())
        m = m.add(m_add)
        return m


class hbPass(nn.Module):
    def __init__(
        self,
        input_channels,
        output_channels,
        previous_conv=False,
        size=0,
        kernel_size=-1,
        stride=-1,
        padding=0,
        dropout=0,
        groups=1,
        Linear=False,
        bias=True,
    ):
        super(hbPass, self).__init__()
        ##########################################################
        self.layer_type = "hbPass"
        self.bias = bias
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.Linear = Linear
        self.previous_conv = previous_conv
        ##########################################################
        if Linear == True:
            self.linear = nn.Linear(input_channels, output_channels)
        ##########################################################
        self.binactive = BinActive.apply
        if not self.Linear:
            self.FPconv = nn.Conv2d(
                input_channels,
                output_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=bias,
            )
        ##########################################################
        else:
            self.linear = nn.Linear(input_channels, output_channels, bias=bias)
        ##########################################################

    def forward(
        self,
        x,
        kernel_size=_pair(3),
        dilation=_pair(1),
        padding=_pair(0),
        stride=_pair(1),
    ):
        if not self.Linear:
            x = self.binactive(x)
            x = self.FPconv(x)
        else:
            x = self.binactive(x)
            if self.previous_conv:
                x = x.view(x.size(0), self.input_channels)
            x = self.linear(x)
        return x


def conv3x3(in_planes, out_planes, stride=1):
    return hbPass(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.relu = nn.ReLU(inplace=True)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.conv1(out)
        out = self.relu(out)

        out = self.bn2(out)
        out = self.conv2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = hbPass(inplanes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = hbPass(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.conv3 = hbPass(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.bn1(out)
        out = self.conv1(x)
        out = self.relu(out)

        out = self.bn2(out)
        out = self.conv2(out)
        out = self.relu(out)

        out = self.bn3(out)
        out = self.conv3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class HbNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        # self.layers = [2, 2, 2, 2]
        self.inplanes = 64
        super(HbNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.BatchNorm2d(self.inplanes),
                hbPass(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
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
        x = self.fc(x)

        return x


# model = HbNet(BasicBlock, [2, 2, 2, 2]).cuda()
# tor = torch.randn(32, 3, 224, 224).cuda()
# print(model(tor).size())

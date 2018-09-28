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

binAgg = 16

# For proving angle preservation property
def angle(a, b):
    cos = nn.CosineSimilarity()
    return math.degrees(math.acos(cos(a.view(1, -1),b.view(1, -1))))


def binAbs(input):
    shape   =   input.size()
    binAgg = 16
    restore = 0
    if(len(shape)==4):
        restore     =   input.size()
        input       =   input.reshape(shape[0], -1)
    shape           =   input.size()
    if(len(shape)==2): 
        input       =   input.unsqueeze(1)
    shape = input.size()
    if(shape[-1] > binAgg):
        listmat = list(torch.split(torch.abs(input), binAgg, dim=-1))
        residualmat = listmat[-1]
        splitup = torch.stack(listmat[:-1])
        stackmat = torch.mean(splitup, dim=-1, keepdim=True).repeat(1, 1, 1, listmat[0].size(-1))
        del listmat, splitup
        residualmat = torch.mean(residualmat, dim=-1, keepdim=True).repeat(1, 1, residualmat.size(-1))
        z = list(stackmat)
        stackmat = torch.cat(z, dim=-1)
        output = torch.cat((stackmat, residualmat), dim=-1)
        del stackmat, residualmat
    else:
        # binmat = input.sign()
        stackmat = torch.mean(input, dim=-1, keepdim=True).repeat(1, 1, input.size(-1))
        output = stackmat
    output = torch.squeeze(output) 
    if(restore!=0):
        output = output.reshape(restore)
    return output



class BinActive(Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        shape   =   input.size()
        if(len(shape) == 2):  
            input = input.unsqueeze(1)
        shape   =   input.size()
        if(shape[-1] > binAgg):
            binmat  =   input.sign()
            listmat = list(torch.split(torch.abs(input), binAgg, dim=-1))
            residualmat = listmat[-1]
            splitup = torch.stack(listmat[:-1])
            stackmat = torch.mean(splitup, dim=-1, keepdim=True).repeat(1, 1, 1, listmat[0].size(-1))
            residualmat = torch.mean(residualmat, dim=-1, keepdim=True).repeat(1, 1, residualmat.size(-1))
            z = list(stackmat)
            stackmat = torch.cat(z, dim=-1)
            output = torch.cat((stackmat, residualmat), dim=-1)
            output = torch.squeeze(output.mul(binmat))
        else:
            binmat = input.sign()
            stackmat = torch.mean(input, dim=-1, keepdim=True).repeat(1, 1, input.size(-1))
            output = stackmat
            output = torch.squeeze(output.mul(binmat))
        return output
    @staticmethod
    def backward(ctx, grad_output):
        input,             =    ctx.saved_tensors
        # output             =    binAbs(input)
        binAgg             =    16
        grad_input         =    grad_output.clone()
        # g_out              =    grad_output.clone()
        # s                  =    g_out.size()
        grad_input[input.le(-1.0)] = 0
        grad_input[input.ge( 1.0)] = 0
        # m = output.abs().mul(grad_input)
        ## Here, div(g_out.nelement()) should be binAgg by proof. 
        ## but that does not work.
        # m_add = (g_out.mul(input.sign())).sum().div(g_out.nelement()).expand(s)
        # m_add = (g_out.mul(input.sign())).sum().div(g_out.nelement()).expand(s)
        # m = m.add(m_add)
        m = grad_input
        return m


class Im2Col(Function):
    @staticmethod
    def forward(ctx, input, kernel_size, dilation, padding, stride):
        assert input.dim() == 4
        ctx.kernel_size = kernel_size
        ctx.dilation = dilation
        ctx.padding = padding
        ctx.stride = stride
        ctx.input_size = (input.size(2), input.size(3))
        ctx._backend = type2backend[input.type()]
        output = input.new()
        ctx._backend.Im2Col_updateOutput(ctx._backend.library_state,        \
                                         input, output,                     \
                                         kernel_size[0], kernel_size[1],    \
                                         dilation[0], dilation[1],          \
                                         padding[0], padding[1],            \
                                         stride[0], stride[1])              
        return output
    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        grad_input = grad_output.new()
        ctx._backend.Im2Col_updateGradInput(ctx._backend.library_state,             \
                                            grad_output,                            \
                                            grad_input,                             \
                                            ctx.input_size[0], ctx.input_size[1],   \
                                            ctx.kernel_size[0], ctx.kernel_size[1], \
                                            ctx.dilation[0], ctx.dilation[1],       \
                                            ctx.padding[0], ctx.padding[1],         \
                                            ctx.stride[0], ctx.stride[1])           
        return grad_input, None, None, None, None

class Col2Im(Function):
    @staticmethod
    def forward(ctx, input, kernel_size, dilation, padding, stride):
        img_h = (torch.sqrt(torch.FloatTensor([input.size(-1)])) - 1)*stride[0] - 2*padding[0] + kernel_size[0]
        output_size = (img_h, img_h)
        ctx.output_size = output_size
        ctx.kernel_size = kernel_size
        ctx.dilation = dilation
        ctx.padding = padding
        ctx.stride = stride
        ctx._backend = type2backend[input.type()]
        output = input.new()
        ctx._backend.Col2Im_updateOutput(ctx._backend.library_state,
                                         input, output,
                                         int(output_size[0].item()), int(output_size[1].item()),
                                         kernel_size[0], kernel_size[1],
                                         dilation[0], dilation[1],
                                         padding[0], padding[1],
                                         stride[0], stride[1])
        return output
    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        grad_input = grad_output.new()
        ctx._backend.Col2Im_updateGradInput(ctx._backend.library_state,
                                            grad_output,
                                            grad_input,
                                            ctx.kernel_size[0], ctx.kernel_size[1],
                                            ctx.dilation[0], ctx.dilation[1],
                                            ctx.padding[0], ctx.padding[1],
                                            ctx.stride[0], ctx.stride[1])
        return grad_input, None, None, None, None, None


class hbPass(nn.Module):
    def __init__(self, input_channels, output_channels,                     \
            kernel_size=-1, stride=-1, padding=-1, dropout=0, groups=1, Linear=False, bias=True):
        super(hbPass, self).__init__()
        ##########################################################
        self.layer_type     =   'hbPass'
        self.kernel_size    =   kernel_size
        self.stride         =   stride
        self.padding        =   padding
        self.Linear         =   Linear
        self.binagg         =   16
        self.bias           =   bias
        ##########################################################
        self.dropout_ratio  =   dropout
        if dropout!=0:
            self.dropout    =   nn.Dropout(dropout)    
        ##########################################################
        self.binactive      =   BinActive.apply
        if not self.Linear:
            self.FPconv     =   nn.Conv2d(input_channels, output_channels,
                                kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=bias)
            self.im2col     =   Im2Col(kernel_size=_pair(kernel_size), dilation=_pair(1), padding=_pair(padding), stride=_pair(stride))
            self.col2im     =   Col2Im(kernel_size=_pair(kernel_size), dilation=_pair(1), padding=_pair(padding), stride=_pair(stride))
            self.bn         =   nn.BatchNorm2d(input_channels, eps=1e-4, momentum=0.1, affine=True)
        ##########################################################
        else:
            self.bn         =   nn.BatchNorm1d(input_channels, eps=1e-4, momentum=0.1, affine=True)
            self.linear         =   nn.Linear(input_channels, output_channels)
        ##########################################################
        self.relu = nn.ReLU(inplace = True)
        ##########################################################
    def forward(self, x, kernel_size=_pair(3), dilation=_pair(1), padding=_pair(0), stride=_pair(1)):
        if not self.Linear:
            x = self.bn(x)
            ##########################################################
            x               =   self.im2col.apply(x, _pair(kernel_size), _pair(dilation), _pair(padding), _pair(stride))
            x               =   self.binactive(x)
            x               =   self.col2im.apply(x, _pair(kernel_size), _pair(dilation), _pair(padding), _pair(stride))
            if self.dropout_ratio!=0:
                x           =   self.dropout(x)
            x               =   self.FPconv(x)
            ##########################################################
        else:
            ##########################################################
            x               =   self.binactive(x)
            x               =   self.linear(x)
            ##########################################################
        x = self.relu(x)
        return x

class hbLin(nn.Module):
    def __init__(self, input_channels, output_channels, dropout=0, bias=True):
        super(hbLin, self).__init__()
        ##########################################################
        self.layer_type     =   'hbLin'
        self.binagg         =   16
        self.bias           =   bias
        ##########################################################
        self.dropout_ratio  =   dropout
        if dropout!=0:
            self.dropout    =   nn.Dropout(dropout)    
        ##########################################################
        self.linear         =   nn.Linear(input_channels, output_channels)
        ##########################################################
    def forward(self, x):    
        ##########################################################
        x               =   self.binactive(x)
        x               =   self.linear(x)
        ##########################################################
        return x



class hbConv(nn.Module):
    def __init__(self, input_channels, output_channels,                     \
            kernel_size=-1, stride=-1, padding=0, dropout=0, groups=1, bias=True):
        super(hbConv, self).__init__()
        ##########################################################
        self.layer_type     =   'hbConv'
        self.kernel_size    =   kernel_size
        self.stride         =   stride
        self.padding        =   padding
        self.binagg         =   16
        self.bias           =   bias
        ##########################################################
        self.dropout_ratio  =   dropout
        if dropout!=0:
            self.dropout    =   nn.Dropout(dropout)    
        ##########################################################
        self.binactive      =   BinActive.apply
        self.FPconv     =   nn.Conv2d(input_channels, output_channels,
                            kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=bias)
        self.im2col     =   Im2Col(kernel_size=_pair(kernel_size), dilation=_pair(1), padding=_pair(padding), stride=_pair(stride))
        self.col2im     =   Col2Im(kernel_size=_pair(kernel_size), dilation=_pair(1), padding=_pair(padding), stride=_pair(stride))
        ##########################################################
    def forward(self, x, kernel_size=_pair(3), dilation=_pair(1), padding=_pair(0), stride=_pair(1)):
        ##########################################################
        x               =   self.im2col.apply(x, _pair(kernel_size), _pair(dilation), _pair(padding), _pair(stride))
        x               =   self.binactive(x)
        x               =   self.col2im.apply(x, _pair(kernel_size), _pair(dilation), _pair(padding), _pair(stride))
        x               =   self.FPconv(x)
        ##########################################################
        return x



'''
    For hbPass
        if Convolutional
            BN --> IM2COL --> BINACTIVE --> COL2IM --> DROPOUT --> CONVOLUTION --> RELU
        if Linear
            BINACTIVE --> LINEAR --> RELU
'''



def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return hbConv(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


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
        self.conv1 = hbConv(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = hbConv(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = hbConv(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
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
    def __init__(self, block, layers, num_classes=10):
        self.inplanes = 64
        super(HbNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool= nn.AvgPool2d(4, stride=1)
        self.fc     = nn.Linear(512 * block.expansion, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                hbConv(self.inplanes, planes * block.expansion,
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
        # x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def hbnet(pretrained=False, **kwargs):
    """Constructs a HbNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = HbNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    # if pretrained:
    return model


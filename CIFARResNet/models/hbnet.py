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
        stackmat = torch.mean(torch.abs(input), dim=-1, keepdim=True).repeat(1, 1, input.size(-1))
        output = stackmat
    output = torch.squeeze(output) 
    if(restore!=0):
        output = output.reshape(restore)
    return output


def eqn_grad_sum(input):
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
        listmat = list(torch.split(input, binAgg, dim=-1))
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

# class BinActive(Function):
#     @staticmethod
#     def forward(ctx, input):
#         ctx.save_for_backward(input)
#         shape   =   input.size()
#         if(len(shape) == 2):  
#             input = input.unsqueeze(1)
#         shape   =   input.size()
#         if(shape[-1] > binAgg):
#             binmat  =   input.sign()
#             listmat = list(torch.split(torch.abs(input), binAgg, dim=-1))
#             residualmat = listmat[-1]
#             splitup = torch.stack(listmat[:-1])
#             stackmat = torch.mean(splitup, dim=-1, keepdim=True).repeat(1, 1, 1, listmat[0].size(-1))
#             residualmat = torch.mean(residualmat, dim=-1, keepdim=True).repeat(1, 1, residualmat.size(-1))
#             z = list(stackmat)
#             stackmat = torch.cat(z, dim=-1)
#             output = torch.cat((stackmat, residualmat), dim=-1)
#             output = torch.squeeze(output.mul(binmat))
#         else:
#             binmat = input.sign()
#             stackmat = torch.mean(torch.abs(input), dim=-1, keepdim=True).repeat(1, 1, input.size(-1))
#             output = stackmat
#             output = torch.squeeze(output.mul(binmat))
#         return output
#     @staticmethod
#     def backward(ctx, grad_output):
#         input,             =    ctx.saved_tensors
#         output             =    binAbs(input)
#         binAgg             =    16
#         grad_input         =    grad_output.clone()
#         g_out              =    grad_output.clone()
#         s                  =    g_out.size()
#         grad_input[input.le(-1.0)] = 0
#         grad_input[input.ge( 1.0)] = 0
#         m = output.abs().mul(grad_input)
#         ## Here, div(g_out.nelement()) should be binAgg by proof. 
#         ## but that does not work.
#         m_add = eqn_grad_sum(g_out.mul(input.sign()))
#         # m_add = (g_out.mul(input.sign())).sum().div(g_out.nelement()).expand(s)
#         # m_add = (g_out.mul(input.sign())).sum().div(g_out.nelement()).expand(s)
#         m = m.add(m_add)
#         # m = grad_input
#         return m
class BinActive(Function): 
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        shape  = input.size()
        output = (input.sign()).mul(input.abs().mean())
        return output
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input1 = grad_output.clone()
        grad_input2 = grad_output.clone()
        grad_input1[input.le(-1.0)] = 0
        grad_input1[input.ge( 1.0)] = 0
        m2 = grad_input1.mul(input.abs().mean())
        m1 = input.sign().div(input.nelement()).mul((grad_input2.mul(input.sign())).sum())
        return m1.add(m2) 


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
            kernel_size=-1, stride=-1, padding=-1, dropout=0, groups=1, Linear=False):
        super(hbPass, self).__init__()
        ##########################################################
        self.layer_type     =   'hbPass'
        self.kernel_size    =   kernel_size
        self.stride         =   stride
        self.padding        =   padding
        self.Linear         =   Linear
        self.binagg         =   16
        ##########################################################
        self.dropout_ratio  =   dropout
        if dropout!=0:
            self.dropout    =   nn.Dropout(dropout)    
        ##########################################################
        self.binactive      =   BinActive.apply
        if not self.Linear:
            self.FPconv     =   nn.Conv2d(input_channels, output_channels,
                                kernel_size=kernel_size, stride=stride, padding=padding, groups=groups)
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



class hbLin(nn.Module):
    def __init__(self, input_channels, output_channels, dropout=0, bias=True):
        super(hbLin, self).__init__()
        ##########################################################
        self.layer_type     =   'hbLin'
        self.binagg         =   16
        self.bias           =   bias
        self.binactive = BinActive.apply
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



'''
    For hbPass
        if Convolutional
            BN --> IM2COL --> BINACTIVE --> COL2IM --> DROPOUT --> CONVOLUTION --> RELU
        if Linear
                BINACTIVE --> LINEAR --> RELU
'''

  #   Preprocessing:
  #       Global contrast normalziation
  #       ZCA whitening
  #       No data-augmentation
  #   Structure:
  # [ ]  128C3 -> ReLU -> 128C3 -> ReLU -> MP2 -> 256C3 -> ReLU  ->
  #            256C3 -> ReLU -> MP2 -> 512C3 -> ReLU -> 512C3 -> 
  #            ReLU -> MP2 -> FC(1024) -> ReLU -> FC(1024) -> ReLU -> 10SVM
  #   MP2 -> Max Pool 2x2
  #   BN with batch size 50
  #   500 epochs.
class HbNet(nn.Module):
    def __init__(self):
        super(HbNet, self).__init__()
        self.xnor = nn.Sequential(
                nn.Conv2d(3, 192, kernel_size=5, stride=1, padding=2),
                nn.BatchNorm2d(192, eps=1e-4, momentum=0.1, affine=False),
                nn.ReLU(inplace=True),
                hbPass(192, 160, kernel_size=1, stride=1, padding=0),
                hbPass(160,  96, kernel_size=1, stride=1, padding=0),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

                hbPass( 96, 192, kernel_size=5, stride=1, padding=2, dropout=0.5),
                hbPass(192, 192, kernel_size=1, stride=1, padding=0),
                hbPass(192, 192, kernel_size=1, stride=1, padding=0),
                nn.AvgPool2d(kernel_size=3, stride=2, padding=1),

                hbPass(192, 192, kernel_size=3, stride=1, padding=1, dropout=0.5),
                hbPass(192, 192, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(192, eps=1e-4, momentum=0.1, affine=False),
                nn.Conv2d(192, 10, kernel_size=1, stride=1, padding=0),
                nn.ReLU(inplace=True),
                nn.AvgPool2d(kernel_size=8, stride=1, padding=0)
                )
    def forward(self, x):
        x = self.xnor(x)
        x = x.view(x.size(0), 10)
        return x
# class HbNet(nn.Module):
#     def __init__(self):
#         super(HbNet, self).__init__()
#         self.features = nn.Sequential(
#                 nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
#                 nn.ReLU(inplace=True),
#                 nn.MaxPool2d(kernel_size=2, stride=2),
#                 nn.BatchNorm2d(64,  eps=1e-4, momentum=0.1, affine=True),
#                 hbConv(64, 192, kernel_size=5, stride=1, padding=2),
#                 nn.ReLU(inplace=True),
#                 nn.MaxPool2d(kernel_size=2, stride=2),
#                 nn.BatchNorm2d(192,  eps=1e-4, momentum=0.1, affine=True),
#                 hbConv(192, 384, kernel_size=3, stride=1, padding=1),
#                 nn.ReLU(inplace=True),
#                 nn.BatchNorm2d(384,  eps=1e-4, momentum=0.1, affine=True),
#                 hbConv(384, 256, kernel_size=3, stride=1, padding=1),
#                 nn.ReLU(inplace=True),
#                 nn.BatchNorm2d(256,  eps=1e-4, momentum=0.1, affine=True),
#                 hbConv(256, 256, kernel_size=3, stride=1, padding=1),
#                 nn.ReLU(inplace=True),
#                 nn.MaxPool2d(kernel_size=2, stride=2),
#                 )
#         self.classifier = nn.Sequential(
#                 nn.Dropout(),
#                 nn.BatchNorm1d(256*2*2, eps=1e-4, momentum=0.1, affine=True),
#                 hbLin(256*2*2, 4096),
#                 nn.ReLU(inplace=True),
#                 nn.Dropout(),
#                 nn.BatchNorm1d(4096, eps=1e-4, momentum=0.1, affine=True),
#                 hbLin(4096, 4096),
#                 nn.ReLU(inplace=True),
#                 nn.Linear(4096, 10),
#             )
#     def forward(self, x):
#         x = self.features(x)
#         # print(x.size())
#         x = x.view(x.size(0), -1)
#         x = self.classifier(x)
#         return x
# model = HbNet().cuda()
# tor = torch.randn(32, 3, 32, 32).cuda()
# print(model(tor).size())

def hbnet(pretrained=False, **kwargs):
    model = HbNet()
    return model
# model = HbNet()
# model.cuda()

# k = model(torch.randn(32, 3, 32, 32).cuda())
# print(k.size())



# class HbNet(nn.Module):
#     def __init__(self):
#         super(HbNet, self).__init__()
#         self.xnor = nn.Sequential(
#                 nn.Conv2d(3, 192, kernel_size=5, stride=1, padding=2),
#                 nn.BatchNorm2d(192, eps=1e-4, momentum=0.1, affine=False),
#                 nn.ReLU(inplace=True),
#                 hbPass(192, 160, kernel_size=1, stride=1, padding=0),
#                 hbPass(160,  96, kernel_size=1, stride=1, padding=0),
#                 nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

#                 hbPass( 96, 192, kernel_size=5, stride=1, padding=2, dropout=0.5),
#                 hbPass(192, 192, kernel_size=1, stride=1, padding=0),
#                 hbPass(192, 192, kernel_size=1, stride=1, padding=0),
#                 nn.AvgPool2d(kernel_size=3, stride=2, padding=1),

#                 hbPass(192, 192, kernel_size=3, stride=1, padding=1, dropout=0.5),
#                 hbPass(192, 192, kernel_size=1, stride=1, padding=0),
#                 nn.BatchNorm2d(192, eps=1e-4, momentum=0.1, affine=False),
#                 nn.Conv2d(192, 10, kernel_size=1, stride=1, padding=0),
#                 nn.ReLU(inplace=True),
#                 nn.AvgPool2d(kernel_size=8, stride=1, padding=0)
#                 )
#     def forward(self, x):
#         x = self.xnor(x)
#         x = x.view(x.size(0), 10)
#         return xnor

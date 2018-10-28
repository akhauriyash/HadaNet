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
    return math.degrees(math.acos(cos(a.view(1, -1),b.view(1, -1))))

def binFunc(input, abs_bin=False, signed_bin=False, binmat_mul=False):
    binAgg = 4
    shape   = input.size()
    restore = 0
    if(len(shape)==4):
        restore   = input.size()
        input     = input.reshape(shape[0], -1)
    shape   = input.size()
    if(len(shape)==2):
        input     = input.unsqueeze(1)
    shape   = input.size()
    if(shape[-1] > binAgg):
        if(abs_bin==True):
            listmat = list(torch.split(torch.abs(input), binAgg, dim=-1))
        elif(binmat_mul==True):
            listmat = list(torch.split(torch.abs(input), binAgg, dim=-1))
            binmat  = input.sign()
        elif(signed_bin==True):
            listmat = list(torch.split(input, binAgg, dim=-1))
        residualmat = torch.mean(listmat[-1], dim=-1, keepdim=True).repeat(1, 1, listmat[-1].size(-1))
        listmat = torch.stack(listmat[:-1])
        listmat = torch.mean(listmat, dim=-1, keepdim=True).repeat(1, 1, 1, listmat[0].size(-1))
        listmat = torch.cat(list(listmat), dim=-1)
        output = torch.cat((listmat, residualmat), dim=-1)
        if(binmat_mul==True):
            output = output.mul(binmat)
    else:
        if(abs_bin==True):
            listmat = list(input)
        elif(binmat_mul==True):
            listmat = list(input)
            binmat = input.sign()
        elif(signed_bin==True):
            listmat = list(input)
        listmat = torch.stack(listmat)
        listmat = torch.mean(listmat, dim=-1, keepdim=True).repeat(1, 1, listmat[-1].size(-1))
        output = listmat
        if(binmat_mul==True):
            output = output.mul(binmat)
    output = torch.squeeze(output)
    if(restore!=0):
        output = output.reshape(restore)
    return output

# class BinActive(Function):
#     @staticmethod
#     def forward(ctx, input):
#         ctx.save_for_backward(input)
#         return input.sign()
#     @staticmethod
#     def backward(ctx, grad_output):
#         input,     =    ctx.saved_tensors
#         grad_input = grad_output.clone()
#         grad_input[input.ge(1)] = 0
#         grad_input[input.le(-1)] = 0
#         return grad_input
        
class BinActive(Function):
    @staticmethod
    def forward(ctx, input):
        output  = binFunc(input, abs_bin=True)
        ctx.save_for_backward(input, Variable(output))
        output = output.mul(input.sign())
        return output
    @staticmethod
    def backward(ctx, grad_output):
        input, output      =    ctx.saved_tensors
        # output             =    binFunc(input, abs_bin=True)
        binAgg             =    4
        grad_input         =    grad_output.clone()
        g_out              =    grad_output.clone()
        s                  =    g_out.size()
        grad_input[input.le(-1.0)] = 0
        grad_input[input.ge( 1.0)] = 0
        m = output.abs().mul(grad_input)
        m_add = grad_output.mul(input.sign())
        m_add = binFunc(m_add, signed_bin=True).mul(input.sign())
        # if len(s) == 4:
        #     m_add = m_add.sum(3, keepdim=True)\
        #             .sum(2, keepdim=True).sum(1, keepdim=True).div(m_add[0].nelement()).expand(s)
        # elif len(s) == 2:
        #     m_add = m_add.sum(1, keepdim=True).div(m_add[0].nelement()).expand(s)
        # m_add = m_add.mul(input.sign())
        m = m.add(m_add)
        return m


# class BinActive(Function):
#     @staticmethod
#     def forward(ctx, input):
#         ctx.save_for_backward(input)
#         output = binFunc(input, binmat_mul=True)
#         return output
#     @staticmethod
#     def backward(ctx, grad_output):
#         input,             =    ctx.saved_tensors
#         output             =    binFunc(input, abs_bin=True)
#         binAgg             =    8
#         grad_input         =    grad_output.clone()
#         g_out              =    grad_output.clone()
#         s                  =    g_out.size()
#         grad_input[input.le(-1.0)] = 0
#         grad_input[input.ge( 1.0)] = 0
#         m = output.abs().mul(grad_input)
#         m_add = grad_output.mul(input.sign())
#         m_add = binFunc(m_add, signed_bin=True).mul(input.sign())
#         # if len(s) == 4:
#         #     m_add = m_add.sum(3, keepdim=True)\
#         #             .sum(2, keepdim=True).sum(1, keepdim=True).div(m_add[0].nelement()).expand(s)
#         # elif len(s) == 2:
#         #     m_add = m_add.sum(1, keepdim=True).div(m_add[0].nelement()).expand(s)
#         m = m.add(m_add)
#         return m

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
    def __init__(self, input_channels, output_channels, previous_conv = False, size=0,\
            kernel_size=-1, stride=-1, padding=-1, dropout=0, groups=1, Linear=False):
        super(hbPass, self).__init__()
        ##########################################################
        self.layer_type     =   'hbPass'
        self.input_channels =   input_channels
        self.output_channels=   output_channels
        self.kernel_size    =   kernel_size
        self.stride         =   stride
        self.padding        =   padding
        self.Linear         =   Linear
        self.previous_conv  =   previous_conv
        self.binagg         =   4
        ##########################################################
        self.dropout_ratio  =   dropout
        if dropout!=0:
            self.dropout    =   nn.Dropout(dropout)    
        if Linear == True:
            self.linear         =   nn.Linear(input_channels, output_channels)
        ##########################################################
        self.binactive      =   BinActive.apply
        if not self.Linear:
            self.FPconv     =   nn.Conv2d(input_channels, output_channels,
                                kernel_size=kernel_size, stride=stride, padding=padding, groups=groups)
            self.im2col     =   Im2Col(kernel_size=_pair(kernel_size), dilation=_pair(1), padding=_pair(padding), stride=_pair(stride))
            self.col2im     =   Col2Im(kernel_size=_pair(kernel_size), dilation=_pair(1), padding=_pair(padding), stride=_pair(stride))
            # self.bn         =   nn.BatchNorm2d(input_channels, eps=1e-4, momentum=0.1, affine=True)
        ##########################################################
        else:
            # if self.previous_conv:
            #     self.bn = nn.BatchNorm2d(input_channels/size, eps=1e-4, momentum=0.1, affine=True)
            # else:
            #     self.bn = nn.BatchNorm1d(input_channels, eps=1e-4, momentum=0.1, affine=True)
            self.linear = nn.Linear(input_channels, output_channels)
        ##########################################################
        self.relu = nn.ReLU(inplace = True)
        ##########################################################
    def forward(self, x, kernel_size=_pair(3), dilation=_pair(1), padding=_pair(0), stride=_pair(1)):
        if not self.Linear:
            # x = self.bn(x)
            ##########################################################
            x               =   self.im2col.apply(x, _pair(kernel_size), _pair(dilation), _pair(padding), _pair(stride))
            x               =   self.binactive(x)
            x               =   self.col2im.apply(x, _pair(kernel_size), _pair(dilation), _pair(padding), _pair(stride))
            if self.dropout_ratio!=0:
                x           =   self.dropout(x)
            x               =   self.FPconv(x)
            ##########################################################
        else:
            x = self.binactive(x)
            if self.previous_conv:
                x = x.view(x.size(0), self.input_channels)
            ##########################################################
            x               =   self.linear(x)
            ##########################################################
        x = self.relu(x)
        return x


'''
    For hbPass
        if Convolutional
            BN --> IM2COL --> BINACTIVE --> COL2IM --> DROPOUT --> CONVOLUTION --> RELU
        if Linear
            BINACTIVE --> LINEAR --> RELU
'''
class HbNet(nn.Module):
    def __init__(self):
        super(HbNet, self).__init__()
        self.conv0 = nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=0)
        self.bn0   = nn.BatchNorm2d(96, eps=1e-4, momentum=0.1, affine=True)
        self.relu0 = nn.ReLU()
        self.mpool0= nn.MaxPool2d(kernel_size=3, stride=2)
        self.bn1   = nn.BatchNorm2d(96, eps=1e-4, momentum=0.1, affine=True)
        self.conv1 = hbPass(96, 256, kernel_size=5, stride=1, padding=2)
        # self.relu1 = nn.ReLU()
        self.mpool1= nn.MaxPool2d(kernel_size=3, stride=2)
        self.bn2   = nn.BatchNorm2d(256, eps=1e-4, momentum=0.1, affine=True)
        self.conv2 = hbPass(256, 384, kernel_size=3, stride=1, padding=1)
        # self.relu2 = nn.ReLU()
        self.bn3   = nn.BatchNorm2d(384, eps=1e-4, momentum=0.1, affine=True)
        self.conv3 = hbPass(384, 384, kernel_size=3, stride=1, padding=1)
        # self.relu3 = nn.ReLU()
        self.bn4   = nn.BatchNorm2d(384, eps=1e-4, momentum=0.1, affine=True)
        self.conv4 = hbPass(384, 256, kernel_size=3, stride=1, padding=1)
        # self.relu4 = nn.ReLU()
        self.mpool4= nn.MaxPool2d(kernel_size=3, stride=2)
        self.bn_c2l= nn.BatchNorm1d(9216, eps=1e-4, momentum=0.1, affine=True)
        self.fc1   = hbPass(256*6*6, 4096, Linear=True, previous_conv=True, size=6*6)
        # self.frelu1= nn.ReLU() 
        self.dropo1= nn.Dropout(0.5)
        self.bn_l2l= nn.BatchNorm1d(4096, eps=1e-4, momentum=0.1, affine=True)
        self.fc2   = hbPass(4096, 4096, Linear=True)
        # self.frelu2= nn.ReLU() 
        self.bn_l2f= nn.BatchNorm1d(4096, eps=1e-4, momentum=0.1, affine=True)
        self.dropo2= nn.Dropout(0.5)
        self.fc3   = nn.Linear(4096, 1000)
        # self.rF    = nn.ReLU()
        # self.fcf   = nn.Linear(1000, 1)
        #dropout -> bn -> conv/inear -> relu
    def forward(self, x):
        # print(x.size())
        x = self.conv0(x)
        x = self.bn0(x)
        x = self.relu0(x)
        # print(x.size())
        x = self.mpool0(x)
        # print(x.size())
        x = self.bn1(x)
        x = self.conv1(x)
        # print(x.size())
        # x = self.relu1(x)
        x = self.mpool1(x)
        # print(x.size())
        x = self.bn2(x)
        x = self.conv2(x)
        # print(x.size())
        # x = self.relu2(x)
        x = self.bn3(x)
        x = self.conv3(x)
        # print(x.size())
        # x = self.relu3(x)
        x = self.bn4(x)
        x = self.conv4(x)
        # print(x.size())
        # x = self.relu4(x)
        x = self.mpool4(x)
        # print(x.size())
        x = x.view(x.size(0), 256*6*6)
        x = self.bn_c2l(x)
        x = self.fc1(x)
        # print(x.size())
        # x = self.frelu1(x)
        x = self.dropo1(x)
        x = self.bn_l2l(x)
        x = self.fc2(x)
        # print(x.size())
        # x = self.frelu2(x)
        x = self.bn_l2f(x)
        x = self.dropo2(x)
        x = self.fc3(x)
        # print(x.size())
        # x = self.rF(x)
        # x = self.fcf(x)
        return x
model = HbNet().cuda()
k = torch.randn(32, 3, 227, 227).cuda()
a = time.time()
print(time.time() - a)
print(model(k).size())
a = time.time()
print(time.time() - a)
print(model(k).size())
a = time.time()
print(time.time() - a)
print(model(k).size())
a = time.time()
print(time.time() - a)
print(model(k).size())
a = time.time()
print(time.time() - a)
print(model(k).size())
# class HbNet(nn.Module):
#     def __init__(self):
#         super(HbNet, self).__init__()
#         self.conv0 = nn.Conv2d(3, 128, kernel_size=3, stride=1, padding=1)
#         self.relu0 = nn.ReLU(inplace=True)
#         self.bn0   = nn.BatchNorm2d(128, eps=1e-4, momentum=0.1, affine=True)
#         self.conv1 = hbPass(128, 128, kernel_size=3, stride=1, padding=1)
#         self.mp1   = nn.MaxPool2d(kerenl_size=2, stride=2)
#         self.bn1   = nn.BatchNorm2d(128, eps=1e-4, momentum=0.1, affine=True)
#         self.conv2 = hbPass(128, 256, kernel_size=2, stride=1, padding=1)
#         self.bn2   = nn.BatchNorm2d(256, eps=1e-4, momentum=0.1, affine=True)
#         self.conv3 = hbPass(256, 256, kernel_size=2, stride=1, padding=1)
#         self.mp2   = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.bn3   = nn.BatchNorm2d(256, eps=1e-4, momentum=0.1, affine=True)
#         self.conv4 = hbPass(256, 512, kernel_size=2, stride=1, padding=1)
#         self.bn4   = nn.BatchNorm2d(512, eps=1e-4, momentum=0.1, affine=True)
#         self.conv5 = hbPass(512, 512, kernel_size=2, stride=1, padding=1)
#         self.mp3   = nn.MaxPool2d(kernel_size=2, stride=1)
#         self.bn_c2l= nn.BatchNorm2d(512, eps=1e-4, momentum=0.1, affine=True)
#         self.fc1   = hbPass(512*10*10, 1024, Linear=True, previous_conv=True, size = 10*10)
#         self.bn_l2l= nn.BatchNorm1d(1024, eps=1e-4, momentum=0.1, affine=True)
#         self.fc2   = hbPass(1024, 1024, Linear=True, previous_conv=False)
#         self.bn_l2F= nn.BatchNorm1d(1024, eps=1e-4, momentum=0.1, affine=True)
#         self.fc3   = nn.Linear(1024, 10)
#         # self.bn_out= nn.BatchNorm1d(10, eps=1e-4, momentum=0.1, affine=True)
#     def forward(self, x):
#         # for m in self.modules():
#         #     if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
#         #         if hasattr(m.weight, 'data'):
#         #             m.weight.data.clamp_(min=0.01)
#         x = self.conv0(x)
#         x = self.relu0(x)
#         x = self.bn0(x)
#         x = self.conv1(x)
#         x = self.mp1(x) 
#         x = self.bn1(x)
#         x = self.conv2(x)
#         x = self.bn2(x)
#         x = self.conv3(x)
#         x = self.mp2(x)
#         x = self.bn3(x)
#         x = self.conv4(x)
#         x = self.bn4(x)
#         x = self.conv5(x)
#         x = self.mp3(x)
#         x = self.bn_c2l(x)
#         x = self.fc1(x)
#         x = self.bn_l2l(x)
#         x = self.fc2(x)
#         x = self.bn_l2F(x)
#         x = self.fc3(x)
#         # x = self.bn_out(x)
#         return x
# model = HbNet().cuda()
# tor = torch.randn(3, 3, 32, 32).cuda()
# print(model(tor).size())
# class HbNet(nn.Module):
#     def __init__(self):
#         super(HbNet, self).__init__()
#         '''
#         Conv -> ReLU -> Pool -> BN -> BinActive() -> Conv -> ReLU
#              -> MaxPool -> BN -> BinActive() -> Lin -> ReLU -> BN 
#              -> BinActive() -> Lin -> ReLU -> BN -> Lin -> SoftMax
#         '''
#         # self.conv1 = nn.Conv2d(1, 20, kernel_size=5, stride=1)
#         # self.bn_conv1 = nn.BatchNorm2d(20, eps=1e-4, momentum=0.1, affine=False)
#         # self.relu_conv1 = nn.ReLU(inplace=True)
#         # self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
#         # self.bin_conv2 = hbPass(20, 50, kernel_size=5, stride=1, padding=0)
#         # self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
#         # self.bin_ip1 = hbPass(50*4*4, 500, Linear=True,
#         #         previous_conv=True, size=4*4)
#         # self.ip2 = nn.Linear(500, 10)
#         # self.bn_c2l = nn.BatchNorm2d(50, eps=1e-4, momentum=0.1, affine=True)
#         # self.bn_l2l = nn.BatchNorm1d(500, eps=1e-4, momentum=0.1, affine=True)
#         self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=1, padding=0)
#         self.relu  = nn.ReLU(inplace=True)
#         self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) 
#         self.bn0   = nn.BatchNorm2d(6, eps=1e-4, momentum=0.1, affine=True)
#         self.conv2 = hbPass(6, 16, kernel_size=5, stride=1, padding=0)
#         self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.bn_c2l = nn.BatchNorm2d(16, eps=1e-4, momentum=0.1, affine=True)
#         self.bin_ipl = hbPass(16*5*5,120, Linear=True, previous_conv=True, size=5*5)
#         self.bn_l2l1 = nn.BatchNorm1d(120, eps=1e-4, momentum=0.1, affine=True)
#         self.ip2   = hbPass(120, 84, Linear=True, previous_conv=False)
#         self.bn_l2l2 = nn.BatchNorm1d(84, eps=1e-4, momentum=0.1, affine=True)
#         self.ip3   = nn.Linear(84, 10)
#         # self.softmax = nn.Softmax()
#         # for m in self.modules():
#         #     if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
#         #         if hasattr(m.weight, 'data'):
#         #             m.weight.data.zero_().add_(1.0)
#         return

#     def forward(self, x):
#         for m in self.modules():
#             if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
#                 if hasattr(m.weight, 'data'):
#                     m.weight.data.clamp_(min=0.01)
#         x = self.conv1(x)
#         x = self.relu(x)
#         x = self.pool1(x)
#         x = self.bn0(x)
#         x = self.conv2(x)
#         x = self.pool2(x)
#         x = self.bn_c2l(x)
#         x = self.bin_ipl(x)
#         x = self.bn_l2l1(x)
#         x = self.ip2(x)
#         x = self.bn_l2l2(x)
#         x = self.ip3(x)
#         # x = self.softmax(x)
#         return x



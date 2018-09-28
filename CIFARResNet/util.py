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



def binActive(input):
    binAgg          =   16
    shape           =   input.size()
    restore         =   0
    if(len(shape)  == 4):
        restore     =   input.size()
        input       =   input.reshape(shape[0], -1)
    shape           =   input.size()
    if(len(shape)  == 2): 
        input       =   input.unsqueeze(1)
    shape           =   input.size()
    if(shape[-1] > binAgg):
        binmat      =   input.sign()
        listmat     =   list(torch.split(torch.abs(input), binAgg, dim=-1))
        residualmat =   listmat[-1]
        splitup     =   torch.stack(listmat[:-1])
        stackmat    =   torch.mean(splitup, dim=-1, keepdim=True).repeat(1, 1, 1, listmat[0].size(-1))
        del listmat, splitup
        residualmat =   torch.mean(residualmat, dim=-1, keepdim=True).repeat(1, 1, residualmat.size(-1))
        stackmat    =   list(stackmat)
        stackmat    =   torch.cat(stackmat, dim=-1)
        output      =   torch.cat((stackmat, residualmat), dim=-1)
        del stackmat, residualmat
        output      =   torch.squeeze(output.mul(binmat))
        del binmat
    else:
        binmat = input.sign()
        stackmat = torch.mean(torch.abs(input), dim=-1, keepdim=True).repeat(1, 1, input.size(-1))
        output = stackmat
        output      = torch.squeeze(output.mul(binmat))
    if(restore != 0):
        output      =   output.reshape(restore)
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


class BinOp():
    def __init__(self, model):
        count_Layers = 0
        for m in model.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                count_Layers  = count_Layers + 1
        start_range           = 1
        end_range             = count_Layers - 2
        self.bin_range        = np.linspace(start_range,
                end_range, end_range-start_range+1)\
                                .astype('int').tolist()
        self.num_of_params    = len(self.bin_range)
        self.saved_params     = []
        self.target_params    = []
        self.target_modules   = []
        index                 = -1
        for m in model.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                index         = index + 1
                if index in self.bin_range:
                    tmp       = m.weight.data.clone()
                    self.saved_params.append(tmp)
                    self.target_modules.append(m.weight)


    def binarization(self):
        self.meanCenterConvParams()
        self.clampConvParams()
        self.save_params()
        self.binarizeConvParams()


    def meanCenterConvParams(self):
        for index in range(self.num_of_params):
            s       = self.target_modules[index].data.size()
            negMean = self.target_modules[index].data.mean(1, keepdim=True).\
                    mul(-1).expand_as(self.target_modules[index].data)
            self.target_modules[index].data = self.target_modules[index].data.add(negMean)
    

    def clampConvParams(self):
        for index in range(self.num_of_params):
            self.target_modules[index].data = \
                    self.target_modules[index].data.clamp(-1.0, 1.0)


    def save_params(self):
        for index in range(self.num_of_params):
            self.saved_params[index].copy_(self.target_modules[index].data)


    def binarizeConvParams(self):
        for index in range(self.num_of_params):
            n      =   self.target_modules[index].data[0].nelement()
            s      =   self.target_modules[index].data.size()
            m      =   binActive(self.target_modules[index].data)
            self.target_modules[index].data = m

 
    def restore(self):
        for index in range(self.num_of_params):
            self.target_modules[index].data.copy_(self.saved_params[index])


    def updateBinaryGradWeight(self):
        for index in range(self.num_of_params):
            weight = self.target_modules[index].data
            binAgg = 16
            n      = weight[0].nelement()
            s      = weight.size()
            m      = binAbs(weight)
            m[weight.lt(-1.0)] = 0 
            m[weight.gt(1.0)] = 0
            m      = m.mul(self.target_modules[index].grad.data)
            m_add  = weight.sign().mul(self.target_modules[index].grad.data)
            # m_add  = m_add.sum(3, keepdim=True)\
                    # .sum(2, keepdim=True).sum(1, keepdim=True).div(binAgg).expand(s)
            # m_add = m_add.sum().div(m_add.nelement()).expand(s)
            m_add = eqn_grad_sum(m_add)
            m_add  = m_add.mul(weight.sign())
            self.target_modules[index].grad.data = m.add(m_add).mul(1.0-1.0/s[1]).mul(1e+8)



# def binActive(input):
#     binAgg  =   16
#     shape   =   input.size()
#     restore = 0
#     if(len(shape)==4):
#         restore     =   input.size()
#         input       =   input.reshape(shape[0], -1)
#     shape           =   input.size()
#     if(len(shape)==2): 
#         input       =   input.unsqueeze(1)
#     shape = input.size()
#     if(shape[-1] > binAgg):
#         binmat  =   input.sign()
#         listmat = list(torch.split(torch.abs(input), binAgg, dim=-1))
#         residualmat = listmat[-1]
#         splitup = torch.stack(listmat[:-1])
#         stackmat = torch.mean(splitup, dim=-1, keepdim=True).repeat(1, 1, 1, listmat[0].size(-1))
#         residualmat = torch.mean(residualmat, dim=-1, keepdim=True).repeat(1, 1, residualmat.size(-1))
#         z = list(stackmat)
#         stackmat = torch.cat(z, dim=-1)
#         output = torch.cat((stackmat, residualmat), dim=-1)
#         output = torch.squeeze(output.mul(binmat))
#     else:
#         binmat = input.sign()
#         listmat = list(torch.split(torch.abs(input), binAgg, dim=-1))
#         splitup = torch.stack(listmat)
#         stackmat = torch.mean(splitup, dim=-1, keepdim=False).repeat(1, 1, 1, listmat[0].size(-1))
#         z = list(stackmat)
#         output = torch.cat(z, dim=-1)
#         output = torch.squeeze(output.mul(binmat))
#     output = torch.squeeze(output) 
#     if(restore!=0):
#         output = output.reshape(restore)
#     return output

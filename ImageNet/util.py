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


class BinOp():
    def __init__(self, model):
        count_Conv2d = 0
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                count_Conv2d += 1
        count_Lin    = count_Conv2d
        for m in model.modules():
            if isinstance(m, nn.Linear):
                count_Lin    += 1

        start_range           = 1
        end_range             = count_Lin - 2

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
                index          += 1
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
            m      =   binFunc(self.target_modules[index].data, binmat_mul=True)
            self.target_modules[index].data = m

 
    def restore(self):
        for index in range(self.num_of_params):
            self.target_modules[index].data.copy_(self.saved_params[index])


    def updateBinaryGradWeight(self):
        for index in range(self.num_of_params):
            weight = self.target_modules[index].data
            binAgg = 4
            n      = weight[0].nelement()
            s      = weight.size()
            m      = binFunc(weight, abs_bin=True)
            m[weight.lt(-1.0)] = 0 
            m[weight.gt(1.0)] = 0
            m      = m.mul(self.target_modules[index].grad.data)
            m_add  = weight.sign().mul(self.target_modules[index].grad.data)
            m_add  = binFunc(m_add, signed_bin=True)
            # if len(s) == 4:
            #     m_add = m_add.sum(3, keepdim=True)\
            #             .sum(2, keepdim=True).sum(1, keepdim=True).div(m_add[0].nelement()).expand(s)
            # elif len(s) == 2:
            #     m_add = m_add.sum(1, keepdim=True).div(m_add[0].nelement()).expand(s)
            m_add  = m_add.mul(weight.sign())
            self.target_modules[index].grad.data = m.add(m_add).mul(1.0-1.0/s[1]).mul(1e+9)

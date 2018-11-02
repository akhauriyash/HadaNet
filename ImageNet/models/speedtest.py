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


tor = torch.randn(64, 224, 224).cuda()
a = time.time()
print(binFunc(tor, abs_bin=True).size())
print(time.time() - a)
tor+=2
a = time.time()
print(binFunc(tor, abs_bin=True).size())
print(time.time() - a)
tor+=2
a = time.time()
print(binFunc(tor, abs_bin=True).size())
print(time.time() - a)
tor+=2
a = time.time()
print(binFunc(tor, abs_bin=True).size())
print(time.time() - a)
tor+=2
a = time.time()
print(binFunc(tor, abs_bin=True).size())
print(time.time() - a)
tor+=2
a = time.time()
print(binFunc(tor, abs_bin=True).size())
print(time.time() - a)
tor+=2
a = time.time()
print(binFunc(tor, abs_bin=True).size())
print(time.time() - a)
tor+=2
a = time.time()
print(binFunc(tor, abs_bin=True).size())
print(time.time() - a)
tor+=2
a = time.time()
print(binFunc(tor, abs_bin=True).size())
print(time.time() - a)
tor+=2
a = time.time()
print(binFunc(tor, abs_bin=True).size())
print(time.time() - a)
tor+=2
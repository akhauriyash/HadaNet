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
import time
import math
import sys
import os
import torch
import argparse
import torchvision
import torchvision.transforms as transforms
import data as dd
import util
import torch.nn as nn
import torch.optim as optim
from models import hbnet
from collections import OrderedDict
from torch.autograd import Variable

        
def pearson(output, original):
    x = output
    y = original
    vx = x - torch.mean(x)
    vy = y - torch.mean(y)
    return torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))


class NetExtract(nn.Module):
    def __init__(self, num):
        super(NetExtract, self).__init__()
        self.features = nn.Sequential(
            *list(original_model.features.children())[:-num]
        )
    def forward(self, x):
        x = self.features(x)
        return x


trainset = dd.dataset(root='./data/', train=True)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
        shuffle=True, num_workers=2)

for i, data in enumerate(trainloader):
    if(i==0):
        input, labels = data
        input = Variable(input).cuda()
    else:
        break

holder = np.empty([3, 256])

layer = 3

# Full precision results
model = hbnet.HbNet(1).cuda()
pmod2 = torch.load('models/hbnet.best.pth.tar')
pmod  = torch.load('models/test.best.pth.tar')
for key, value in pmod2['state_dict'].items():
    pmod['state_dict'][key[7:]] = value
model.load_state_dict(pmod['state_dict'])

original = nn.Sequential(*list(model.features.children())[:-layer])(input)

i = 0

for Ba in range(1, 17):
    # Initialize model
    model = hbnet.HbNet(Ba).cuda()
    # Load model weights
    pmod2 = torch.load('models/hbnet.best.pth.tar')
    pmod  = torch.load('models/test.best.pth.tar')
    for key, value in pmod2['state_dict'].items():
        pmod['state_dict'][key[7:]] = value
    model.load_state_dict(pmod['state_dict'])
    #Loop through different weight binarization aggressions
    for Bw in range(1, 17):
        bin_op = util.BinOp(model, Bw)
        bin_op.binarization()
        output = nn.Sequential(*list(model.features.children())[:-layer])(input)
        cost = pearson(output, original)
        holder[:, :, i] = [Ba, Bw, cost]
        i+=1


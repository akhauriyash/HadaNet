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
import matplotlib.pyplot as plt
from matplotlib.mlab import griddata
import torchvision.transforms as transforms
import data as dd
import util
import torch.nn as nn
import torch.optim as optim
from models import hbnet
from collections import OrderedDict
from torch.autograd import Variable
from matplotlib import ticker, cm
from mpl_toolkits.mplot3d import Axes3D
def pearson(output, original):
    x = output
    y = original
    vx = x - torch.mean(x)
    vy = y - torch.mean(y)
    return torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))


trainset = dd.dataset(root='./data/', train=True)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
        shuffle=True, num_workers=2)

for i, data in enumerate(trainloader):
    if(i==0):
        input, labels = data
        input = Variable(input).cuda()
    else:
        break

holder = np.empty([3, 64])

for layer in range(29):
    # Full precision results
    model = hbnet.HbNet(1).cuda()
    pmod2 = torch.load('models/hbnet.best.pth.tar')
    pmod  = torch.load('models/test.best.pth.tar')
    for key, value in pmod2['state_dict'].items():
        pmod['state_dict'][key[7:]] = value
    model.load_state_dict(pmod['state_dict'])

    bin_op = util.BinOp(model, 1)
    bin_op.binarization()
    
    original = nn.Sequential(*list(model.children()))[0][:-layer](input)
    print(layer, "\t", nn.Sequential(*list(model.children()))[0][-layer])
    # print(original.size())
    i = 0
    for Ba in range(1, 9):
        # Initialize model
        model = hbnet.HbNet(Ba).cuda()
        # Load model weights
        pmod2 = torch.load('models/hbnet.best.pth.tar')
        pmod  = torch.load('models/test.best.pth.tar')
        for key, value in pmod2['state_dict'].items():
            pmod['state_dict'][key[7:]] = value
        model.load_state_dict(pmod['state_dict'])
        #Loop through different weight binarization aggressions
        for Bw in range(1, 9):
            bin_op = util.BinOp(model, Bw)
            bin_op.binarization()
            output = nn.Sequential(*list(model.children()))[0][:-layer](input)
            cost = pearson(output, original)
            holder[:, i] = [Ba, Bw, cost]
            i+=1

    x=holder[0, :] 
    y=holder[1, :] 
    z=holder[2, :] 

    xi = np.linspace(x.min()-1, x.max()+1, 64)
    yi = np.linspace(y.min()-1, y.max()+1, 64)
    zi = griddata(x, y, z, xi, yi, interp='linear')
    fig, ax = plt.subplots()
    CS = plt.contourf(xi, yi, zi, 8, cmap = cm.Greys)
    plt.xlabel('Ba')
    title = "Layer " + str(layer)
    plt.title(title)
    plt.ylabel('Bw')
    plt.xlim([1,8])
    plt.ylim([1,8])
    cbar = fig.colorbar(CS)
    name = 'l' + str(layer) + '.eps'
    plt.savefig(name)

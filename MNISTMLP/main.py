from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import sys
import os
import torch
import argparse
import torchvision
import torchvision.transforms as transforms
import util
import torch.nn as nn
import torch.optim as optim
from models import hbnet
from torch.autograd import Variable

def save_state(model, optimizer, acc):
    print("==> Saving model ...")
    state = {
            'acc'          : acc,
            'state_dict' : model.state_dict(),
            'optimizer'  : optimizer.state_dict(),
            }
    torch.save(state, 'models/hbnet.best.pth.tar')


def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(trainloader):
        # Binarization of weights
        bin_op.binarization()
        # Forward pass
        data, target = Variable(data.cuda()), Variable(target.cuda())
        optimizer.zero_grad()
        output          = model(data)
        # Backward pass
        ## L2-SVM etc.?
        loss         = criterion(output, target)
        loss.backward()
        # Restore full precision weights
        bin_op.restore()
        bin_op.updateBinaryGradWeight()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [({:.2f}%)]\tLoss: {:.4f}\tLR: {}'.format(
                epoch,
                100*batch_idx / len(trainloader), loss.data[0],
                optimizer.param_groups[0]['lr']))
    return

def test():
    global best_acc
    model.eval()
    test_loss = 0
    correct   = 0
    bin_op.binarization()
    for data, target in testloader:
        data, target = Variable(data.cuda()), Variable(target.cuda())
        output       = model(data)
        test_loss   += criterion(output, target).data[0]
        pred         = output.data.max(1, keepdim=True)[1]
        correct     += pred.eq(target.data.view_as(pred)).cpu().sum()
    correct          = float(correct)
    bin_op.restore()
    acc              = 100*float(correct)/float(len(testloader.dataset))
    test_loss        /= float(len(testloader.dataset))

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({}%)'.format(
        test_loss * 128, correct, len(testloader.dataset),
        100*float(correct)/float(len(testloader.dataset))))
    
    print('Best Accuracy: {}%\n'.format(float(best_acc)))

    if acc > best_acc:
        best_acc     = acc
        save_state(model, optimizer, best_acc)
    return

def adjust_learning_rate(optimizer, epoch):
    update_list = [120, 200, 240, 280]
    if epoch in update_list:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.1
    return

if __name__=='__main__':
    cpu         =    False
    data        =    './data'
    arch        =    'hbnet'
    lr          =    0.01
    pretrained  =    False
    evaluate    =    False


    # Seed for reproducibility
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)

    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])


    trainset    = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

    testset     = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader  = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)

    # define classes
    classes     = ('plane', 'car', 'bird', 'cat',
                'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    # define the model
    print('==> building model ...')
    if arch    == 'hbnet':
        model   = hbnet.HbNet()
    else:
        raise Exception(arch+' is currently not supported')

    base_lr     = float(lr)
    param_dict  = dict(model.named_parameters())
    params      = []

    for key, value in param_dict.items():
        params += [{'params'      : [value],
                    'lr'          :    base_lr,
                    'weight_decay': 0.00001}]
        optimizer = optim.Adam(params, lr=0.050, weight_decay=0.00001)
    criterion   = nn.CrossEntropyLoss()

    ## MODEL INITIALIZATION
    if not pretrained:
        print('==> Initializing model parameters ...')
        best_acc = 0
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.05)
                m.bias.data.zero_()
    ## Model loading
    else:
        print('==> Load pretrained model form', pretrained, '...')
        pmod2 = torch.load('models/hbnet.best.pth.tar')
        pmod  = torch.load('models/test.best.pth.tar')
        for key, value in pmod2['state_dict'].items():
            pmod['state_dict'][key[7:]] = value
        model.load_state_dict(pmod['state_dict'])
        best_acc = pmod['acc']

    if not cpu:
        model.cuda()
        model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    print(model)
    if not pretrained:
        print("Skipping optimizer loading")
    else:
           optimizer.load_state_dict(pmod2['optimizer'])
       
        if evaluate:
            test()
            exit(0)

        for epoch in range(0, 320):
           adjust_learning_rate(optimizer, epoch)
           train(epoch)
           test(optimizer)
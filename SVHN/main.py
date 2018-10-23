from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import time
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
from collections import OrderedDict
from torch.autograd import Variable

def save_state(model, optimizer, best_acc):
    print("==> Saving model ...")
    state = {
            'acc'        : best_acc,
            'state_dict' : model.state_dict(),
            'optimizer'  : optimizer.state_dict(),
            }            
    test=0
    if(test==1):
        state['state_dict'] = OrderedDict([(k.replace('module.', ''), v) if 'module' in k else (k,v) for k, v in state['state_dict'].items()])
        torch.save(state, 'models/test.best.pth.tar')
    else:
        torch.save(state, 'models/hbnet.pth.tar')
    # state = {
    #         'acc'        : best_acc,
    #         'state_dict' : model.state_dict(),
    #         'optimizer'  : optimizer.state_dict(),
    #         }            
    # state['state_dict'] = OrderedDict([(k.replace('module.', ''), v) if 'module' in k else (k,v) for k, v in state['state_dict'].items()])
    # torch.save(state, 'models/test.best.pth.tar')
def save_state2(model, optimizer, best_acc):
    print("==> Saving model ...")
    state = {
            'acc'        : best_acc,
            'state_dict' : model.state_dict(),
            'optimizer'  : optimizer.state_dict(),
            }            
    test=0
    if(test==1):
        state['state_dict'] = OrderedDict([(k.replace('module.', ''), v) if 'module' in k else (k,v) for k, v in state['state_dict'].items()])
        torch.save(state, 'models/test.best.pth.tar')
    else:
        torch.save(state, 'models/hbnet2.pth.tar')
    # state = {
    #         'acc'        : best_acc,
    #         'state_dict' : model.state_dict(),
    #         'optimizer'  : optimizer.state_dict(),
    #         }            
    # state['state_dict'] = OrderedDict([(k.replace('module.', ''), v) if 'module' in k else (k,v) for k, v in state['state_dict'].items()])
    # torch.save(state, 'models/test.best.pth.tar')

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
    update_list = [10, 20, 50, 100, 120]
    if epoch in update_list:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.1
    return

if __name__=='__main__':
    cpu         =    False
    data        =    './data'
    arch        =    'hbnet'
    lr          =    0.05
    pretrained  =    True
    evaluate    =    False


    # Seed for reproducibility
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)

    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])


    trainset    = torchvision.datasets.SVHN(root='./data', split='train', download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=192, shuffle=True, num_workers=2)

    testset     = torchvision.datasets.SVHN(root='./data', split='test', download=True, transform=transform_test)
    testloader  = torch.utils.data.DataLoader(testset, batch_size=192, shuffle=False, num_workers=2)

    # define classes
    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # define the model
    # print('==> building model',arch,'...')
    if arch == 'hbnet':
        model = hbnet.HbNet()
    else:
        raise Exception(arch+' is currently not supported')

    # initialize the model

    if not pretrained:
        print('==> Initializing model parameters ...')
        best_acc = 0
        for m in model.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
#                m.weight.data.normal_(0, 0.05)
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data = m.weight.data.zero_().add(1.0)
    else:
        # print('==> Load pretrained model form', pretrained, '...')
        pmod2 = torch.load('models/hbnet.pth.tar')
        # print(pmod['state_dict'].keys())
        print("****************************")
        pmod = torch.load('models/test.best.pth.tar')
        # print(pmod['state_dict'].keys())
        # print(len(pmod2['state_dict'].keys()))
        # print(len(pmod['state_dict'].keys()))
        for key, value in pmod2['state_dict'].items():
            pmod['state_dict'][key[7:]] = value
        # print(len(pmod2['state_dict'].keys()))
        # print(len(pmod['state_dict'].keys()))
        # model.load_state_dict(pmod['state_dict'])
        best_acc = 0
        model.load_state_dict(pmod['state_dict'])

    if not cpu:
        model.cuda()
        model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    print(model)

    # define solver and criterion
    base_lr = float(lr)
    param_dict = dict(model.named_parameters())
    params = []

    for key, value in param_dict.items():
        params += [{'params':[value], 'lr': base_lr,
            'weight_decay':0.00001}]

        optimizer = optim.Adam(params, lr=0.10,weight_decay=0.00001)
    criterion = nn.CrossEntropyLoss()
#    if pretrained:
 #       optimizer.load_state_dict(pmod2['optimizer'])
    # define the binarization operator
    bin_op = util.BinOp(model)
    # save_state(model, 0)
    # do the evaluation if specified
    if evaluate:
        test()
        exit(0)
    #save_state(model, optimizer, 0)
    #print("stop now")
    #time.sleep(4)
    # start training
    for epoch in range(1, 320):
        if(epoch%10 == 0):
            save_state2(model, optimizer, 0)
        adjust_learning_rate(optimizer, epoch)
        train(epoch)
        test()

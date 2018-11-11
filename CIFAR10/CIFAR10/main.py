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
import data as dd
import util
import torch.nn as nn
import torch.optim as optim
from models import hbnet
from collections import OrderedDict
from torch.autograd import Variable

def save_state(model, optimizer, acc):
    print("==> Saving model ...")
    state = {
            'acc'        : acc,
            'state_dict' : model.state_dict(),
            'optimizer'  : optimizer.state_dict(),
            }            
    test=0
    if(test==0):
        state['state_dict'] = OrderedDict([(k.replace('module.', ''), v) if 'module' in k else (k,v) for k, v in state['state_dict'].items()])
        torch.save(state, 'models/test.best.pth.tar')
    else:
        torch.save(state, 'models/hbnet.best.pth.tar')

def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(trainloader):
        # Binarization of weights
        bin_op.binarization()
        # Forward pass
        data, target = Variable(data.cuda()), Variable(target.cuda())
        optimizer.zero_grad()
        output       = model(data)
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
    test_loss       /= len(testloader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({}%)'.format(
        test_loss , correct, len(testloader.dataset),
        100*float(correct)/float(len(testloader.dataset))))
    
    print('Best Accuracy: {}%\n'.format(float(best_acc)))

    if acc > best_acc:
        best_acc     = acc
        save_state(model, optimizer, best_acc)
    return

# def adjust_learning_rate(optimizer, epoch):
#     update_list = [60, 120, 200, 240, 280]
#     if epoch in update_list:
#         for param_group in optimizer.param_groups:
#             param_group['lr'] = param_group['lr'] * 0.2
#     # lr_start = 0.003
#     # lr_fin   = 0.000002
#     # lr_decay = (lr_fin/lr_start)**(1./epoch)
#     # for param_group in optimizer.param_groups:
#     #     param_group['lr'] = lr_decay
    # return
def adjust_learning_rate(optimizer, epoch):
    if epoch < 30:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.01
        return 0.01
    elif epoch < 60:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.001
        return 0.001
    elif epoch < 100:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.0001
        return 0.0001
    elif epoch < 150:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.00001
        return 0.00001
    elif epoch < 200:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.000001
        return 0.000001
    else:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.0000001
        return 0.0000001
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
    # transform_train = transforms.Compose([
    #     transforms.RandomCrop(32, padding=4),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    # ])

    # transform_test = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    # ])


    # trainset    = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    # trainloader = torch.utils.data.DataLoader(trainset, batch_size=192, shuffle=True, num_workers=2)

    # testset     = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    # testloader  = torch.utils.data.DataLoader(testset, batch_size=192, shuffle=False, num_workers=2)

    trainset = dd.dataset(root='./data/', train=True)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=192,
            shuffle=True, num_workers=2)

    testset = dd.dataset(root='./data/', train=False)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100,
            shuffle=False, num_workers=2)

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
                    'lr'          : base_lr,
                    'weight_decay': 0.00001}]
    # optimizer = optim.SGD(params, lr=.1, momentum=0.9, nesterov=True)
    optimizer = optim.Adam(params, lr=0.1, weight_decay=0.00001)
    # optimizer = torch.optim.Adagrad(params, lr=1.0, rho=0.9, eps=1e-06, weight_decay=1e-5)

    criterion   = nn.CrossEntropyLoss()
  #  acc = 88.27
  #  state = {
  #                      'acc'        : acc,
  #                                  'state_dict' : model.state_dict(),
  #                                              'optimizer'  : optimizer.state_dict(),
  #                                                          }            
  #  state['state_dict'] = OrderedDict([(k.replace('module.', ''), v) if 'module' in k else (k,v) for k, v in state['state_dict'].items()])
  #  torch.save(state, 'models/test.best.pth.tar')

    ## MODEL INITIALIZATION
    if not pretrained:
        print('==> Initializing model parameters ...')
        best_acc = 0
        # for m in model.modules():
        #     if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        #         c = float(m.weight.data[0].nelement())
        #         m.weight.data = m.weight.data.normal_(0, 1.0/c)
        #     # elif isinstance(m, nn.BatchNorm2d):
        #     #     m.weight.data = m.weight.data.zero_().add(1.0)
        for m in model.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                # m.weight.data.normal_(0, 0.05)
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)
            # elif isinstance(m, nn.BatchNorm2d):
            #     m.weight.data = m.weight.data.zero_().add(1.0)
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
    bin_op = util.BinOp(model)
    if not pretrained:
        print("Skipping optimizer loading")
    else:
      pass
        #  optimizer.load_state_dict(pmod2['optimizer'])
      #  for state in optimizer.state.values():
      #      for k, v in state.items():
      #          if isinstance(v, torch.Tensor):
      #              state[k] = v.to('cuda')
  #  save_state(model, optimizer, 0)
 #   print("roko BC")
#    time.sleep(5)
    if evaluate:
        test()
        exit(0)
    # print("SAVING")
    # save_state(model, optimizer, 0)
    # print("STOP NOW")
    # time.sleep(4) 
    # save_state(model, optimizer, 0)
    # print("STOP NOW")
    # time.sleep(5)

    for epoch in range(71, 320):
        if(epoch%10 == 0):
            save_state(model, optimizer, 0)
        lr = adjust_learning_rate(optimizer, epoch)
        train(epoch)
        test()

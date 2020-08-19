from sharpness import cal_sharpness

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import datetime

from models import *
import json

# ckpt = './checkpoint/ckpt1.pth'
# ckpt = './checkpoint/novograd0.05_ckptbest.pth'
ckpt = './checkpoint/sgdwm0.2_ckptbest.pth'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(torch.__version__)

checkpoint = torch.load(ckpt, map_location=lambda storage, loc: storage)
checkpoint['net'] = {k[7:]:checkpoint['net'][k] for k in checkpoint['net']}

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

testset = torchvision.datasets.CIFAR10(
    root='/tmp/cifar10', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False)

net = ResNet50()
net.to(device)
net.load_state_dict(checkpoint['net'])
criterion = nn.CrossEntropyLoss()
res,reseps,resori = cal_sharpness(net,testloader,criterion,[1e-4,1e-5,1e-6,-1e-6,-1e-5,-1e-4])
print(res,reseps,resori)
json.dump([res,reseps,resori],open('sharpness measure'))
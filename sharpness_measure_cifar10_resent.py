from sharpness import cal_sharpness

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
from utils import progress_bar
import argparse
import datetime

from models import *
import json

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 sharpness measure')

parser.add_argument('--resume-best', '-rb', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--resume-worst', '-rw', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--resume-init', '-ri', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
                    help='input batch size for testing (default: 128)')
parser.add_argument('--warmup-epochs', type=int, default=5, metavar='WE',
                    help='number of warmup epochs (default: 5)')
parser.add_argument('--lr-decay', nargs='+', type=int, default=[50, 75],
                    help='epoch intervals to decay lr')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--weight-decay', type=float, default=5e-4, metavar='W',
                    help='SGD weight decay (default: 5e-4)')
parser.add_argument('--optimizer',type=str,default='sgd',
                    help='different optimizers')
parser.add_argument('--lb',action='store_true',
                    help='resume form lb checkpoint')


parser.add_argument('--max-lr',default=0.1,type=float)
parser.add_argument('--div-factor',default=25,type=float)
parser.add_argument('--final-div',default=10000,type=float)

args = parser.parse_args()
if not args.lb:
    ckptbest = './checkpoint/'+args.optimizer+str(args.max_lr)+'_ckptbest.pth'
else:
    ckptbest = './checkpoint/' + args.optimizer +'_lb_'+ str(args.max_lr) + '_ckptbest.pth'

device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(torch.__version__)

net = ResNet50()
net.to(device)

if args.resume_best:
    checkpoint = torch.load(ckptbest, map_location=lambda storage, loc: storage)
    checkpoint['net'] = {k[7:]: checkpoint['net'][k] for k in checkpoint['net']}
    print("best acc",checkpoint['acc'])
    net.load_state_dict(checkpoint['net'])

# elif args.resume_worst:
#     checkpoint = torch.load(ckptworst, map_location=lambda storage, loc: storage)
#     checkpoint['net'] = {k[7:]: checkpoint['net'][k] for k in checkpoint['net']}
#     net.load_state_dict(checkpoint['net'])
# elif args.resume_init:
#     checkpoint = torch.load(ckptinit, map_location=lambda storage, loc: storage)
#     checkpoint['net'] = {k[7:]: checkpoint['net'][k] for k in checkpoint['net']}
#     net.load_state_dict(checkpoint['net'])



transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
trainset = torchvision.datasets.CIFAR10(
    root='/tmp/cifar10', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=args.batch_size, shuffle=True)


testset = torchvision.datasets.CIFAR10(
    root='/tmp/cifar10', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=128, shuffle=True)



criterion = nn.CrossEntropyLoss()

def test(epoch):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    print(correct/total)
def test_on_train(epoch,dataloader):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(dataloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))

res,reseps,resori = cal_sharpness(net,testloader,criterion,[1e-3,5e-4,1e-4,1e-5,1e-6])

json.dump([res,reseps,resori],open('sharpness_measure_{}_{}.json'.format(args.optimizer,args.max_lr),'w+'))
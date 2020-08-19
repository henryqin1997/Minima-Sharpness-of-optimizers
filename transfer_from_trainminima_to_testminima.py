import sharpness

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
from utils import progress_bar
import json

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 train on test')
parser.add_argument('--lr', default=1, type=float, help='learning rate')
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
parser.add_argument('--epoch',type=int,default=30)

parser.add_argument('--max-lr',default=0.1,type=float)
parser.add_argument('--div-factor',default=25,type=float)
parser.add_argument('--final-div',default=10000,type=float)

args = parser.parse_args()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
ckptinit = './checkpoint/'+args.optimizer+str(args.max_lr)+'_ckptinit.pth'
ckptbest = './checkpoint/'+args.optimizer+str(args.max_lr)+'_ckptbest.pth'
ckptworst = './checkpoint/'+args.optimizer+str(args.max_lr)+'_ckptworst.pth'
# Data
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

trainset = torchvision.datasets.CIFAR10(
    root='/tmp/cifar10', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=args.batch_size, shuffle=True)

testset = torchvision.datasets.CIFAR10(
    root='/tmp/cifar10', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False)

net = ResNet50()
net.to(device)

if args.resume_best:
    checkpoint = torch.load(ckptbest, map_location=lambda storage, loc: storage)
    net.load_state_dict(checkpoint['net'])

elif args.resume_worst:
    checkpoint = torch.load(ckptworst, map_location=lambda storage, loc: storage)
    net.load_state_dict(checkpoint['net'])

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.max_lr/args.final_div)

train_on_test_acc = []
valid_on_train_acc = []
train_on_test_loss = []
test_on_train_loss = []

def train_on_test(epoch):
    print('\nTraining on test, Epoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(testloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))
    train_on_test_acc.append(correct / total)
    train_on_test_loss.append(float(train_loss/len(testloader)))

def test_on_train(epoch):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    valid_on_train_acc.append(correct / total)
    test_on_train_loss.append(test_loss/len(trainloader))

for epoch in range(args.epoch):
    train_on_test(epoch)
    test_on_train(epoch)

file = open(args.optimizer+str(args.max_lr)+'-'+datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')+'_onecycle_transfer_train_to test_log.json','w+')
json.dump([train_on_test_acc,valid_on_train_acc,train_on_test_loss,test_on_train_loss],file)


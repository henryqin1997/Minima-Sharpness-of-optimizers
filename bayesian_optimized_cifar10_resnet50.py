'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from bayes_opt import BayesianOptimization

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models import *
from utils import progress_bar,create_lr_scheduler
from torch.optim.lr_scheduler import LambdaLR
import json

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--lr-min', default=1e-5, type=float, help='learning rate')
parser.add_argument('--lr-max', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
                    help='input batch size for testing (default: 128)')
parser.add_argument('--warmup-epochs', type=int, default=5, metavar='WE',
                    help='number of warmup epochs (default: 5)')
parser.add_argument('--lr-decay', nargs='+', type=int, default=[120, 150],
                    help='epoch intervals to decay lr')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--weight-decay', type=float, default=5e-4, metavar='W',
                    help='SGD weight decay (default: 5e-4)')
parser.add_argument('--optimizer',type=str,default='sgd',
                    help='different optimizers')
args = parser.parse_args()

def main(lr=0.1):
    args.lr = lr
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

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
        trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(
        root='/tmp/cifar10', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')

    # Model
    print('==> Building model..')
    # net = VGG('VGG19')
    # net = ResNet18()
    # net = PreActResNet18()
    # net = GoogLeNet()
    # net = DenseNet121()
    # net = ResNeXt29_2x64d()
    # net = MobileNet()
    # net = MobileNetV2()
    # net = DPN92()
    # net = ShuffleNetG2()
    # net = SENet18()
    # net = ShuffleNetV2(1)
    # net = EfficientNetB0()
    # net = RegNetX_200MF()
    net = ResNet50()
    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    ckpt = './checkpoint/'+args.optimizer+str(lr)+'_ckpt.pth'

    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(ckpt)
        net.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']

    criterion = nn.CrossEntropyLoss()
    if args.optimizer.lower()=='sgd':
        optimizer = optim.SGD(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.optimizer.lower()=='sgdwm':
        optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum,
                          weight_decay=args.weight_decay)
    elif args.optimizer.lower()=='adam':
        optimizer = torch.optim.Adam(net.parameters(), lr=args.lr,
                          weight_decay=args.weight_decay)
    elif args.optimizer.lower() == 'rmsprop':
        optimizer = optim.RMSprop(net.parameters(),lr=args.lr, momentum=args.momentum,
                          weight_decay=args.weight_decay)
    elif args.optimizer.lower() == 'adagrad':
        optimizer = optim.Adagrad(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer.lower() == 'radam':
        from radam import RAdam
        optimizer = RAdam(net.parameters(),lr=args.lr,weight_decay=args.weight_decay)
    elif args.optimizer.lower() == 'lars':#no tensorboardX
        from lars import LARS
        optimizer = LARS(net.parameters(), lr=args.lr,momentum=args.momentum,weight_decay=args.weight_decay)
    elif args.optimizer.lower() == 'lamb':
        from lamb import Lamb
        optimizer  = Lamb(net.parameters(),lr=args.lr,weight_decay=args.weight_decay)
    elif args.optimizer.lower() == 'novograd':
        from novograd import NovoGrad
        optimizer = NovoGrad(net.parameters(), lr=args.lr,weight_decay=args.weight_decay)
    else:
        optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum,
                              weight_decay=args.weight_decay)
    # lrs = create_lr_scheduler(args.warmup_epochs, args.lr_decay)
    # lr_scheduler = LambdaLR(optimizer,lrs)
    # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, args.lr_decay, gamma=0.1)
    train_acc = []
    valid_acc = []

    # Training
    def train(epoch):
        print('\nEpoch: %d' % epoch)
        net.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            # lr_scheduler.step()
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
        train_acc.append(correct/total)

    def test(epoch):
        global best_acc
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
        acc = 100.*correct/total
        valid_acc.append(correct/total)

        if acc > best_acc:
            print('Saving..')
            state = {
                'net': net.state_dict(),
                'acc': acc,
                'epoch': epoch,
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, ckpt)
            best_acc = acc

    for epoch in range(200):
        if epoch in args.lr_decay:
            checkpoint = torch.load(ckpt)
            net.load_state_dict(checkpoint['net'])
            best_acc = checkpoint['acc']
            args.lr*=0.1
            if args.optimizer.lower() == 'sgd':
                optimizer = optim.SGD(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            if args.optimizer.lower() == 'sgdwm':
                optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum,
                                      weight_decay=args.weight_decay)
            elif args.optimizer.lower() == 'adam':
                optimizer = optim.Adam(net.parameters(), lr=args.lr,
                                             weight_decay=args.weight_decay)
            elif args.optimizer.lower() == 'rmsprop':
                optimizer = optim.RMSprop(net.parameters(), lr=args.lr, momentum=args.momentum,
                                          weight_decay=args.weight_decay)
            elif args.optimizer.lower() == 'adagrad':
                optimizer = optim.Adagrad(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            elif args.optimizer.lower() == 'radam':
                from radam import RAdam

                optimizer = RAdam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            elif args.optimizer.lower() == 'lars':  # no tensorboardX
                optimizer = LARS(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay,
                                 dampening=args.damping)
            elif args.optimizer.lower() == 'lamb':
                optimizer = Lamb(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            elif args.optimizer.lower() == 'novograd':
                optimizer = NovoGrad(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            else:
                optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum,
                                      weight_decay=args.weight_decay)
        train(epoch)
        test(epoch)
    file = open(args.optimizer+str(lr)+'log.json','w+')
    json.dump([train_acc,valid_acc],file)
    return best_acc

pbounds = {
    'lr': (1e-5, 0.5),
    }

bayes_optimizer = BayesianOptimization(
    f=main,
    pbounds=pbounds,
    verbose=1
)
bayes_optimizer.maximize(init_points=1, n_iter=50)
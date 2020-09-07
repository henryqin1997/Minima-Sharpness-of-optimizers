from helper import *
# from novograd import NovoGrad
from lars import LARS
from lamb import Lamb
from radam import RAdam
from novograd import NovoGrad
# import torch.distributed as dist
import argparse
import sys
import json

model = LeNet5(N_CLASSES)
model = nn.DataParallel(model)
model.to(DEVICE)


parser = argparse.ArgumentParser()
parser.add_argument('-n', '--nodes', default=1,
                    type=int, metavar='N')
parser.add_argument('-g', '--gpus', default=1, type=int,
                    help='number of gpus per node')
parser.add_argument('-nr', '--nr', default=0, type=int,
                    help='ranking within the nodes')
parser.add_argument('--epochs', default=3, type=int,
                    metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-p','--optimizer',default='sgd',type=str,
                    help='optimizer chozen to train')
parser.add_argument('--lr',type = float,default=1, metavar='LR',
                    help='base learning rate (default: 0.1)')
args = parser.parse_args()


if args.optimizer.lower()=='adam':
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
elif args.optimizer.lower()=='sgd':
    optimizer = optim.SGD(model.parameters(), lr=args.lr)
elif args.optimizer.lower()=='sgdwm':
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
elif args.optimizer.lower() == 'rmsprop':
    optimizer = optim.RMSprop(model.parameters(),lr=args.lr, momentum=0.9)
elif args.optimizer.lower() == 'adagrad':
    optimizer = optim.Adagrad(model.parameters(), lr=args.lr)
elif args.optimizer.lower() == 'radam':
    optimizer = RAdam(model.parameters(),lr=args.lr)
elif args.optimizer.lower() == 'lars':#no tensorboardX
    optimizer = LARS(model.parameters(), lr=args.lr, momentum=0.9)
elif args.optimizer.lower() == 'lamb':
    optimizer  = Lamb(model.parameters(),lr=args.lr)
elif args.optimizer.lower() == 'novograd':
    optimizer = NovoGrad(model.parameters(), lr=args.lr, weight_decay=0.0001)
else:
    optimizer = optim.SGD(model.parameters(), lr=1)

optname = args.optimizer if len(sys.argv)>=2 else 'sgd'

# log = open(optname+'log.txt','w+')

def lrs(batch):
    low = 1e-5
    high = 10
    return low + (high - low) * batch / len(train_loader) / args.epochs

lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,lrs)

criterion = nn.CrossEntropyLoss()

loss_list = []

def train(train_loader, model, criterion, optimizer, device, scheduler):
    '''
    Function for the training step of the training loop
    '''

    model.train()
    running_loss = 0

    for X, y_true in train_loader:
        optimizer.zero_grad()

        X = X.to(device)
        y_true = y_true.to(device)

        # Forward pass
        y_hat, _ = model(X)
        loss = criterion(y_hat, y_true)
        loss_list.append(loss.item())
        running_loss += loss.item() * X.size(0)

        # Backward pass
        loss.backward()
        optimizer.step()
        scheduler.step()

    epoch_loss = running_loss / len(train_loader.dataset)
    return model, optimizer, epoch_loss

for i in range(args.epochs):
    train(train_loader,model,criterion,optimizer,'cuda',lr_scheduler)

with open('mnist_lenet_batchsize32_'+args.optimizer+'.json','w+') as f:
    json.dump(loss_list,f)
from helper_onecycle import *
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
parser.add_argument('--epochs', default=2, type=int,
                    metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-p','--optimizer',default='sgd',type=str,
                    help='optimizer chozen to train')
parser.add_argument('--lr',type = float,default=0.01, metavar='LR',
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
    optimizer = optim.SGD(model.parameters(), lr=0.01)

optname = args.optimizer if len(sys.argv)>=2 else 'sgd'

# log = open(optname+'log.txt','w+')

criterion = nn.CrossEntropyLoss()

lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,args.max_lr,steps_per_epoch=len(train_loader),
                                                   epochs=args.epochs)

model, optimizer, _ = training_loop(model, criterion, optimizer, train_loader, valid_loader, N_EPOCHS, DEVICE,lr_scheduler)

with open(optname+str(args.lr)+'_onecycle_loss.txt','w+') as myfile:
    json.dump(_,myfile)
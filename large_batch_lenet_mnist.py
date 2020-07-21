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

BATCH_SIZE = 8192

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
args = parser.parse_args()


if args.optimizer.lower()=='adam':
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
elif args.optimizer.lower()=='sgd':
    optimizer = optim.SGD(model.parameters(), lr=0.01)
elif args.optimizer.lower()=='sgdwm':
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
elif args.optimizer.lower() == 'rmsprop':
    optimizer = optim.RMSprop(model.parameters(),lr=0.001, momentum=0.9)
elif args.optimizer.lower() == 'adagrad':
    optimizer = optim.Adagrad(model.parameters(), lr=0.01)
elif args.optimizer.lower() == 'radam':
    optimizer = RAdam(model.parameters())
elif args.optimizer.lower() == 'lars':#no tensorboardX
    optimizer = LARS(model.parameters(), lr=0.1, momentum=0.9)
elif args.optimizer.lower() == 'lamb':
    optimizer  = Lamb(model.parameters())
elif args.optimizer.lower() == 'novograd':
    optimizer = NovoGrad(model.parameters(), lr=0.01, weight_decay=0.001)
    schedular = optim.lr_scheduler.CosineAnnealingLR(optimizer, 3 * len(train_loader), 1e-4)


    def train(train_loader, model, criterion, optimizer, schedular, device):
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
            running_loss += loss.item() * X.size(0)

            # Backward pass
            loss.backward()
            optimizer.step()
            schedular.step()

        epoch_loss = running_loss / len(train_loader.dataset)
        return model, optimizer, epoch_loss

else:
    optimizer = optim.SGD(model.parameters(), lr=0.01)

optname = args.optimizer if len(sys.argv)>=2 else 'sgd'

# log = open(optname+'log.txt','w+')

log = None

criterion = nn.CrossEntropyLoss()

model, optimizer, _ = training_loop(model, criterion, optimizer, train_loader, valid_loader, N_EPOCHS, DEVICE,log)

with open(optname+'_loss.txt','w+') as myfile:
    json.dump(_,myfile)
from helper import *
# from novograd import NovoGrad
from lars import LARS
from lamb import Lamb
from radam import RAdam
import matplotlib.pyplot as plt
import sys
import json

model = LeNet5(N_CLASSES).to(DEVICE)

if len(sys.argv)==1:
    optimizer = optim.SGD(model.parameters(), lr=0.01)
elif sys.argv[1]=='adam':
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
elif sys.argv[1]=='sgd':
    optimizer = optim.SGD(model.parameters(), lr=0.01)
elif sys.argv[1]=='sgdwm':
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
elif sys.argv[1] == 'rmsprop':
    optimizer = optim.RMSprop(model.parameters(),lr=0.1, momentum=0.9)
elif sys.argv[1] == 'adagrad':
    optimizer = optim.Adagrad(model.parameters(), lr=0.01, lr_decay=0.99, weight_decay=0.9, initial_accumulator_value=0, eps=1e-10)
elif sys.argv[1] == 'radam':
    optimizer = RAdam(model.parameters(), lr=0.001, mom=0.9, sqr_mom=0.99, eps=1e-05, wd=0.0, beta=0.0, decouple_wd=True)
elif sys.argv[1] == 'lars':#no tensorboardX
    optimizer = LARS(model.parameters(), lr=0.1, momentum=0.9)
elif sys.argv[1] == 'lamb':
    optimizer  = Lamb(model.parameters())

# elif sys.argv[1] == 'novograd':
#     optimizer = NovoGrad(model.parameters(), lr=0.01, weight_decay=0.001)
#     schedular = optim.lr_scheduler.CosineAnnealingLR(optimizer, 3 * len(train_loader), 1e-4)
#     exit(0)

else:
    optimizer = optim.SGD(model.parameters(), lr=0.01)

optname = sys.argv[1] if len(sys.argv)>=2 else 'sgd'

log = open(optname+'log.txt','w+')

criterion = nn.CrossEntropyLoss()

model, optimizer, _ = training_loop(model, criterion, optimizer, train_loader, valid_loader, N_EPOCHS, DEVICE,log)

with open(optname+'_loss.txt','w+') as myfile:
    json.dump(_,myfile)
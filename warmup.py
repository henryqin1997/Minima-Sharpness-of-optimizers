from helper import *
import matplotlib.pyplot as plt
import sys
import json

model = LeNet5(N_CLASSES).to(DEVICE)

if len(sys.argv)==1:
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)
elif sys.argv[1]=='adam':
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
elif sys.argv[1]=='sgd':
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)
elif sys.argv[1]=='sgdwm':
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
else:
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)

optname = sys.argv[1] if len(sys.argv)>=2 else 'sgd'

criterion = nn.CrossEntropyLoss()

model, optimizer, _ = training_loop(model, criterion, optimizer, train_loader, valid_loader, N_EPOCHS, DEVICE)

with open(optname+'_loss.txt','w+') as myfile:
    json.dump(_,myfile)
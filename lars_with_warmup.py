from helper import *
# from novograd import NovoGrad
from lars import LARS
from lamb import Lamb
from radam import RAdam
from novograd import NovoGrad
import sys
import json

model = LeNet5(N_CLASSES).to(DEVICE)



optname = 'lars_with_warmup'

log = open(optname+'log.txt','w+')

criterion = nn.CrossEntropyLoss()

def training_loop(model, criterion, train_loader, valid_loader, epochs, device, log, print_every=1):
    '''
    Function defining the entire training loop
    '''

    # set objects for storing metrics
    train_losses = []
    valid_losses = []
    train_accuracy = []
    valid_accuracy = []

    # Train model
    for epoch in range(0, epochs):

        # training
        if epoch<15:
            optimizer = LARS(model.parameters(),lr=0.1*(epoch+1)/15)
        else:
            optimizer = LARS(model.parameters(),lr=0.1*(0.9)**(epoch-15))
        model, optimizer, train_loss = train(train_loader, model, criterion, optimizer, device)
        train_losses.append(train_loss)

        # validation
        with torch.no_grad():
            model, valid_loss = validate(valid_loader, model, criterion, device)
            valid_losses.append(valid_loss)

        if epoch % print_every == (print_every - 1):
            train_acc = get_accuracy(model, train_loader, device=device)
            valid_acc = get_accuracy(model, valid_loader, device=device)
            train_accuracy.append(train_acc)
            valid_accuracy.append(valid_acc)
            log.write(f'{datetime.now().time().replace(microsecond=0)} --- '
                  f'Epoch: {epoch}\t'
                  f'Train loss: {train_loss:.4f}\t'
                  f'Valid loss: {valid_loss:.4f}\t'
                  f'Train accuracy: {100 * train_acc:.2f}\t'
                  f'Valid accuracy: {100 * valid_acc:.2f}\n')

    #plot_losses(train_losses, valid_losses)

    return model, (train_losses, valid_losses)

model, _ = training_loop(model, criterion, train_loader, valid_loader, N_EPOCHS, DEVICE,log)

with open(optname+'_loss.txt','w+') as myfile:
    json.dump(_,myfile)
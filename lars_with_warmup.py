from helper import *
# from novograd import NovoGrad
from lars import LARS
from lamb import Lamb
from radam import RAdam
from novograd import NovoGrad
import sys
import json

model = LeNet5(N_CLASSES).to(DEVICE)

# optimizer = LARS(model.parameters(), lr=0.1, momentum=0.9)

optname = 'lars_with_warmup_momentum'

criterion = nn.CrossEntropyLoss()

def training_loop_lars(model,optimizer, criterion, train_loader, valid_loader, epochs, device, print_every=1):
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

        #training
        if epoch<10:
            optimizer = LARS(model.parameters(),lr=0.1*(epoch+1)/10,momentum=0.9)
        elif epoch<15:
            optimizer = LARS(model.parameters(), lr=0.1,momentum=0.9)
        else:
            optimizer = LARS(model.parameters(),lr=0.1*(0.95**(epoch-15)),momentum=0.9)
        model, optimizer, train_loss = train(train_loader, model, criterion, optimizer, device)
        train_losses.append(train_loss)

        # validation
        with torch.no_grad():
            model, valid_loss = validate(valid_loader, model, criterion, device)
            valid_losses.append(valid_loss)

        if epoch % print_every == (print_every - 1):
            train_acc = get_accuracy(model, train_loader, device=device)
            valid_acc = get_accuracy(model, valid_loader, device=device)
            train_accuracy.append(float(train_acc))
            valid_accuracy.append(float(valid_acc))
            print(f'{datetime.now().time().replace(microsecond=0)} --- '
                  f'Epoch: {epoch}\t'
                  f'Train loss: {train_loss:.4f}\t'
                  f'Valid loss: {valid_loss:.4f}\t'
                  f'Train accuracy: {100 * train_acc:.2f}\t'
                  f'Valid accuracy: {100 * valid_acc:.2f}\n')

    #plot_losses(train_losses, valid_losses)

    return model, [train_losses, valid_losses,train_accuracy,valid_accuracy]

model, _ = training_loop_lars(model,optimizer, criterion, train_loader, valid_loader, N_EPOCHS, DEVICE)

with open(optname+'_loss.txt','w+') as myfile:
    json.dump(_,myfile)
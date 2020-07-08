from matplotlib import pyplot as plt
import re
import json

name_list = ['sgd','sgdwm','rmsprop','adagrad','adam','radam','lars','lamb','novograd']

for name in name_list:
    file = open('loss/'+name+'_loss.txt','r')
    train_loss,valid_loss = json.load(file)
    plt.plot(train_loss,label = name+'_train')
    plt.plot(valid_loss,label = name+'_valid',linestyle='dashed')

plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.show()

for name in name_list:
    file = open('log/'+name+'log.txt','r')
    valid_acc = []
    for line in file:
        valid_acc.append(float(line[-6:]))
    plt.plot(valid_acc,label = name)

plt.xlabel('epoch')
plt.ylabel('valid accuracy')
plt.legend()
plt.show()
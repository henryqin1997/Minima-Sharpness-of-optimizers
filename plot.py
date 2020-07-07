from matplotlib import pyplot as plt
import json

name_list = ['sgd','sgdwm','adagrad','adam','radam','lars','lamb','novograd']

for name in name_list:
    file = open('loss/'+name+'_loss.txt','r')
    train_loss,valid_loss = json.load(file)
    plt.plot(train_loss,label = name+'_train')
    plt.plot(valid_loss,label = name+'_valid',linestyle='dashed')

plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.show()
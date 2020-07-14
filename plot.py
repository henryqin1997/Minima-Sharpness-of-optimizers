from matplotlib import pyplot as plt
import re
import json

name_list = ['sgd','sgdwm','rmsprop','adagrad','adam','radam','lars','lamb','novograd']

color = ['r','g','b','c','y','k','m','violet','orange']

markers = ['.','^','2','s','p','*','+','x','D']

for i,name in enumerate(name_list):
    file = open('loss/'+name+'_loss.txt','r')
    train_loss,valid_loss = json.load(file)
    plt.plot(train_loss,label = name+'_train',color = color[i],marker = markers[i])
    plt.plot(valid_loss,label = name+'_valid',linestyle='dashed',color = color[i],marker = markers[i])

plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.show()

for i,name in enumerate(['novograd','sgd','sgdwm']):
    file = open('loss/'+name+'_loss.txt','r')
    train_loss,valid_loss = json.load(file)
    plt.plot(train_loss,label = name+'_train',color = color[i],marker = markers[i])
    plt.plot(valid_loss,label = name+'_valid',linestyle='dashed',color = color[i],marker = markers[i])

plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.show()

for i,name in enumerate(['novograd','adagrad','rmsprop']):
    file = open('loss/'+name+'_loss.txt','r')
    train_loss,valid_loss = json.load(file)
    plt.plot(train_loss,label = name+'_train',color = color[i],marker = markers[i])
    plt.plot(valid_loss,label = name+'_valid',linestyle='dashed',color = color[i],marker = markers[i])

plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.show()

for i,name in enumerate(['novograd','adam','radam']):
    file = open('loss/'+name+'_loss.txt','r')
    train_loss,valid_loss = json.load(file)
    plt.plot(train_loss,label = name+'_train',color = color[i],marker = markers[i])
    plt.plot(valid_loss,label = name+'_valid',linestyle='dashed',color = color[i],marker = markers[i])

plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.show()

for i,name in enumerate(['novograd','lars','lamb']):
    file = open('loss/'+name+'_loss.txt','r')
    train_loss,valid_loss = json.load(file)
    plt.plot(train_loss,label = name+'_train',color = color[i],marker = markers[i])
    plt.plot(valid_loss,label = name+'_valid',linestyle='dashed',color = color[i],marker = markers[i])

plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.show()
# for name in name_list:
#     file = open('log/'+name+'log.txt','r')
#     valid_acc = []
#     for line in file:
#         valid_acc.append(float(line[-6:]))
#     plt.plot(valid_acc,label = name)
#
# plt.xlabel('epoch')
# plt.ylabel('valid accuracy')
# plt.legend()
# plt.show()
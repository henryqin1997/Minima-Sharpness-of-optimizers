from matplotlib import pyplot as plt
import json
import glob
# name_list = ['sgd','sgdwm','rmsprop','adagrad','adam','radam','lars','lamb','novograd']

#lb mnist lenet
# name_list = ['sgd1.0','sgdwm0.1','rmsprop0.001','adagrad0.01','adam0.01','radam0.1','lars3.0','lamb0.01','novograd0.01']

#sb cifar10 resnet56
name_list = ['*lr0.1_sgd*','*lr0.1_sgdwm*','*lr0.0001_rmsprop*','*lr0.05_adagrad*','*lr0.001_adam*','*lr0.01_radam*','*lr1.0_lars*','*lr0.01_lamb*','*lr0.01_novograd*']

color = ['r','g','b','c','y','k','m','violet','orange']

markers = ['.','^','2','s','p','*','+','x','D']

dir = 'logs/'
# for i,name in enumerate(name_list):
#     file = open('lbloss/'+name+'_loss.txt','r')
#     train_loss,valid_loss,train_accuracy,valid_accuracy = json.load(file)
#     plt.plot(train_loss,label = name+'_train',color = color[i],marker = markers[i])
#     plt.plot(valid_loss,label = name+'_valid',linestyle='dashed',color = color[i],marker = markers[i])
#
# plt.xlabel('epoch')
# plt.ylabel('loss')
# plt.legend()
# plt.show()


#sb mnist-lenet
#
# for i,name in enumerate([name_list[8],name_list[0],name_list[1]]):
#     file = open(dir+name+'_loss.txt','r')
#     train_loss,valid_loss,train_accuracy,valid_accuracy = json.load(file)
#     plt.plot(train_loss,label = name+'_train',color = color[i],marker = markers[i])
#     plt.plot(valid_loss,label = name+'_valid',linestyle='dashed',color = color[i],marker = markers[i])
#
# plt.xlabel('epoch')
# plt.ylabel('loss')
# plt.legend()
# plt.show()
#
# for i,name in enumerate([name_list[8],name_list[3],name_list[2]]):
#     file = open(dir+name+'_loss.txt','r')
#     train_loss, valid_loss, train_accuracy, valid_accuracy = json.load(file)
#     plt.plot(train_loss,label = name+'_train',color = color[i],marker = markers[i])
#     plt.plot(valid_loss,label = name+'_valid',linestyle='dashed',color = color[i],marker = markers[i])
#
# plt.xlabel('epoch')
# plt.ylabel('loss')
# plt.legend()
# plt.show()
#
# for i,name in enumerate([name_list[8],name_list[4],name_list[5]]):
#     file = open(dir+name+'_loss.txt','r')
#     train_loss, valid_loss, train_accuracy, valid_accuracy = json.load(file)
#     plt.plot(train_loss,label = name+'_train',color = color[i],marker = markers[i])
#     plt.plot(valid_loss,label = name+'_valid',linestyle='dashed',color = color[i],marker = markers[i])
#
# plt.xlabel('epoch')
# plt.ylabel('loss')
# plt.legend()
# plt.show()
#
# for i,name in enumerate([name_list[8],name_list[6],name_list[7]]):
#     file = open(dir+name+'_loss.txt','r')
#     train_loss, valid_loss, train_accuracy, valid_accuracy = json.load(file)
#     plt.plot(train_loss,label = name+'_train',color = color[i],marker = markers[i])
#     plt.plot(valid_loss,label = name+'_valid',linestyle='dashed',color = color[i],marker = markers[i])
#
# plt.xlabel('epoch')
# plt.ylabel('loss')
# plt.legend()
# plt.show()


#sb cifar10 resnet56
for i,name in enumerate(name_list):
    fn = glob.glob(dir+name)[0]
    file = open(fn,'r')
    train_loss,train_accuracy,valid_loss,valid_accuracy = json.load(file)
    plt.plot(train_loss,label = name+'_train',color = color[i],marker = markers[i])
    plt.plot(valid_loss,label = name+'_valid',linestyle='dashed',color = color[i],marker = markers[i])

plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.show()

for i,name in enumerate([name_list[8],name_list[0],name_list[1]]):
    fn = glob.glob(dir + name)[0]
    file = open(fn, 'r')
    train_loss, train_accuracy, valid_loss, valid_accuracy = json.load(file)
    plt.plot(train_loss,label = name+'_train',color = color[i],marker = markers[i])
    plt.plot(valid_loss,label = name+'_valid',linestyle='dashed',color = color[i],marker = markers[i])

plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.show()

for i,name in enumerate([name_list[8],name_list[3],name_list[2]]):
    fn = glob.glob(dir + name)[0]
    file = open(fn, 'r')
    train_loss, train_accuracy, valid_loss, valid_accuracy = json.load(file)
    plt.plot(train_loss,label = name+'_train',color = color[i],marker = markers[i])
    plt.plot(valid_loss,label = name+'_valid',linestyle='dashed',color = color[i],marker = markers[i])

plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.show()

for i,name in enumerate([name_list[8],name_list[4],name_list[5]]):
    fn = glob.glob(dir + name)[0]
    file = open(fn, 'r')
    train_loss, train_accuracy, valid_loss, valid_accuracy = json.load(file)
    plt.plot(train_loss,label = name+'_train',color = color[i],marker = markers[i])
    plt.plot(valid_loss,label = name+'_valid',linestyle='dashed',color = color[i],marker = markers[i])

plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.show()

for i,name in enumerate([name_list[8],name_list[6],name_list[7]]):
    fn = glob.glob(dir + name)[0]
    file = open(fn, 'r')
    train_loss, train_accuracy, valid_loss, valid_accuracy = json.load(file)
    plt.plot(train_loss,label = name+'_train',color = color[i],marker = markers[i])
    plt.plot(valid_loss,label = name+'_valid',linestyle='dashed',color = color[i],marker = markers[i])

plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.show()
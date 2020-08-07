from matplotlib import pyplot as plt
import json
import glob
# name_list = ['sgd','sgdwm','rmsprop','adagrad','adam','radam','lars','lamb','novograd']

#lb mnist lenet
# name_list = ['sgd1.0','sgdwm0.1','rmsprop0.001','adagrad0.01','adam0.01','radam0.1','lars3.0','lamb0.01','novograd0.01']

#sb cifar10 resnet56
name_list = ['*lr0.125_sgd*','*lr0.13_sgdwm*','*lr0.00011_rmsprop*','*lr0.03_adagrad*','*lr0.00125_adam*','*lr0.01_radam*','*lr0.4_lars*','*lr0.01_lamb*','*lr0.01_novograd*']

#lb cifar10 resnet56
#name_list = ['*lr0.075_sgd*','*lr0.075_sgdwm*','*lr0.001_rmsprop*','*lr0.01_adagrad*','*lr0.0075_adam*','*lr0.1_radam*','*lr1.5_lars*','*lr0.0125_lamb*','*lr0.005_novograd*']


color = ['r','g','b','c','y','k','m','violet','orange']

markers = ['.','^','2','s','p','*','+','x','D']

dir = 'logs/'
#dir = 'cifar_resnet_lbloss/'

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
from matplotlib import pyplot as plt
import json
train_acc,valid_acc = json.load(open('original_loadbest_multistep_log.json','r'))
plt.plot(valid_acc)
plt.show()

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
# for i,name in enumerate(name_list):
#     fn = glob.glob(dir+name)[0]
#     file = open(fn,'r')
#     train_loss,train_accuracy,valid_loss,valid_accuracy = json.load(file)
#     #plt.plot(train_loss,label = name+'_train',color = color[i],marker = markers[i])
#     #plt.plot(valid_loss,label = name+'_valid',linestyle='dashed',color = color[i],marker = markers[i])
#     plt.plot(valid_accuracy, label=name + '_valid', linestyle='dashed', color=color[i], marker=markers[i])
#
# plt.xlabel('epoch')
# plt.ylabel('validation accuracy')
# plt.legend()
# plt.show()
#
# for i,name in enumerate([name_list[8],name_list[0],name_list[1]]):
#     fn = glob.glob(dir + name)[0]
#     file = open(fn, 'r')
#     train_loss, train_accuracy, valid_loss, valid_accuracy = json.load(file)
#     # plt.plot(train_loss,label = name+'_train',color = color[i],marker = markers[i])
#     # plt.plot(valid_loss,label = name+'_valid',linestyle='dashed',color = color[i],marker = markers[i])
#     plt.plot(valid_accuracy, label=name + '_valid', linestyle='dashed', color=color[i], marker=markers[i])
#
# plt.xlabel('epoch')
# plt.ylabel('validation accuracy')
# plt.legend()
# plt.show()
#
# for i,name in enumerate([name_list[8],name_list[3],name_list[2]]):
#     fn = glob.glob(dir + name)[0]
#     file = open(fn, 'r')
#     train_loss, train_accuracy, valid_loss, valid_accuracy = json.load(file)
#     # plt.plot(train_loss,label = name+'_train',color = color[i],marker = markers[i])
#     # plt.plot(valid_loss,label = name+'_valid',linestyle='dashed',color = color[i],marker = markers[i])
#     plt.plot(valid_accuracy, label=name + '_valid', linestyle='dashed', color=color[i], marker=markers[i])
# plt.xlabel('epoch')
# plt.ylabel('validation accuracy')
# plt.legend()
# plt.show()
#
# for i,name in enumerate([name_list[8],name_list[4],name_list[5]]):
#     fn = glob.glob(dir + name)[0]
#     file = open(fn, 'r')
#     train_loss, train_accuracy, valid_loss, valid_accuracy = json.load(file)
#     # plt.plot(train_loss,label = name+'_train',color = color[i],marker = markers[i])
#     # plt.plot(valid_loss,label = name+'_valid',linestyle='dashed',color = color[i],marker = markers[i])
#     plt.plot(valid_accuracy, label=name + '_valid', linestyle='dashed', color=color[i], marker=markers[i])
#
# plt.xlabel('epoch')
# plt.ylabel('validation accuracy')
# plt.legend()
# plt.show()
#
# for i,name in enumerate([name_list[8],name_list[6],name_list[7]]):
#     fn = glob.glob(dir + name)[0]
#     file = open(fn, 'r')
#     train_loss, train_accuracy, valid_loss, valid_accuracy = json.load(file)
#     # plt.plot(train_loss,label = name+'_train',color = color[i],marker = markers[i])
#     # plt.plot(valid_loss,label = name+'_valid',linestyle='dashed',color = color[i],marker = markers[i])
#     plt.plot(valid_accuracy, label=name + '_valid', linestyle='dashed', color=color[i], marker=markers[i])
#
# plt.xlabel('epoch')
# plt.ylabel('validation accuracy')
# plt.legend()
# plt.show()
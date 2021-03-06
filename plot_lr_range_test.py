from matplotlib import pyplot as plt
import json
import math
# train_acc,valid_acc = json.load(open('original_loadbest_multistep_log.json','r'))
# print(valid_acc.index(max(valid_acc[:150])))
# print((valid_acc.index(max(valid_acc[150:250]))))
# print((valid_acc.index(max(valid_acc[250:]))))

low = math.log2(1e-5)
high = math.log2(10)

# low = math.log2(1e-10)
# high = math.log2(1e-3)



log_neg_one = -3.321928094887362
log_neg_two = -6.643856189774724
log_neg_three = -9.965784284662087

name_list = ['sgd','sgdwm','rmsprop','adagrad','adam','radam','lars','lamb','novograd']

title_list = ['SGD','SGDM','RMSProp','AdaGrad','Adam','RAdam','LARS','LAMB','NovoGrad']

# original curve large batch
for opt in name_list[7:8]:
    # trainloss = json.load(open('lr_range_test_data/'+opt+'_lr_range_find_minibatch.json'))
    trainloss = json.load(open('lbonecyclelog/'+opt+'_batchsize_8192_lr_range_find_minibatch.json'))
    x = [2**(low+(high-low)*i/len(trainloss)) for i in range(int(len(trainloss)*(log_neg_one+1-low)/(high-low)))]
    y = trainloss[:int(len(trainloss)*(log_neg_one+1-low)/(high-low))]
    i = trainloss.index(min(trainloss))
    print(i, 2 ** (low + (high - low) * i / len(trainloss)))
    plt.plot(x,y)
    plt.xscale('log')
    plt.xlabel('learning rate')
    plt.ylabel('loss')
    plt.title('LAMB Lr Range Test')
    plt.show()

#original curve
# for opt in name_list[2:3]:
#     trainloss = json.load(open('lr_range_test_data/'+opt+'_lr_range_find_minibatch.json'))
#     #trainloss = json.load(open(opt+'_batchsize_8192_lr_range_find_minibatch.json'))
#     x = [2**(low+(high-low)*i/len(trainloss)) for i in range(int(len(trainloss)*(log_neg_two-low)/(high-low)))]
#     y = trainloss[:int(len(trainloss)*(log_neg_two-low)/(high-low))]
#     i = trainloss.index(min(trainloss))
#     print(i, 2 ** (low + (high - low) * i / len(trainloss)))
#     plt.plot(x,y)
#     plt.xscale('log')
#     plt.xlabel('learning rate')
#     plt.ylabel('loss')
#     plt.title(opt+'_lr_range_test')
#     plt.show()


#smooth curve
# for opt in name_list[5:6]:
#     trainloss = json.load(open('lr_range_test_data/'+opt+'_lr_range_find_minibatch.json'))
#     trainloss = [sum(trainloss[i-20:i])/20 for i in range(20,int(391*5*(log_neg_one-low)/(high-low)))]
#     x = [2**(low+(high-low)*i/391/5) for i in range(20,int(391*5*(log_neg_one-low)/(high-low)))]
#     y = trainloss
#     i = y.index(min(y))
#     print(i, 2 ** (low + (high - low) * i / 391 / 5))
#     plt.plot(x,y)
#     plt.xscale('log')
#     plt.title(opt+'_lr_range_test')
#     plt.show()


#rate
# for opt in name_list[:1]:
#     trainloss = json.load(open('lr_range_test_data/'+opt+'_lr_range_find_minibatch.json'))
#     x = [2**(low+(high-low)*i/391/5) for i in range(1,int(391*5*(0-low)/(high-low)))]
#     y = [trainloss[i]-trainloss[i-1] for i in range(1,int(391*5*(0-low)/(high-low)))]
#     i = y.index(min(y))
#     print(i, 2 ** (low + (high - low) * i / 391 / 5))
#     plt.plot(x,y)
#     plt.xscale('log')
#     plt.title(opt+'_lr_range_test')
#     plt.show()

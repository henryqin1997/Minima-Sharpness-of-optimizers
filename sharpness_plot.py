import json
from matplotlib import pyplot as plt

# prefix = 'sharpness_measure_'
# postfix = '.json'
# fns = ['sgdwm_0.2','sgdwm_0.03','sgdwm_0.002']
# lb = ['SGDM max-lr-0.2','SGDM max-lr-0.03','SGDM max-lr-0.002']
# x = [5e-4,2e-4,1e-4,5e-5,1e-5,1e-6,0,-1e-6,-1e-5,-5e-5,-2e-4,-1e-4,-5e-4]
# x = [1000*i for i in x]
# x[10],x[11]=x[11],x[10]
# x = x[2:-2]
# color = ['r','g','b']
# for i,fn in enumerate(fns):
#     dif,eps,original,loss,acc = json.load(open(prefix+fn+postfix))
#     y = original[:6]+[loss]+original[6:]
#     y[10],y[11] = y[11],y[10]
#     y = y[2:-2]
#     plt.plot(x,y,label=lb[i]+' acc:'+str(acc),color=color[i])
# plt.xlabel('epsilon (1e-3)')
# plt.ylabel('loss')
# plt.legend()
# plt.title('SGDM sharpness (batchsize 128)')
# plt.show()
#
#
prefix = 'sharpness_measure_'
postfix = '.json'
fns = ['sgdwm_0.2','sgdwm_0.03']
lb = ['SGDM max-lr-0.2','SGDM max-lr-0.03','SGDM max-lr-0.002']
x = [5e-4,2e-4,1e-4,5e-5,1e-5,1e-6,0,-1e-6,-1e-5,-5e-5,-2e-4,-1e-4,-5e-4]
x = [1000*i for i in x]
x[10],x[11]=x[11],x[10]
x = x[0:-1]
color = ['r','g','b']
for i,fn in enumerate(fns):
    dif,eps,original,loss,acc = json.load(open(prefix+fn+postfix))
    y = original[:6]+[loss]+original[6:]
    y[10],y[11] = y[11],y[10]
    y = y[0:-1]
    plt.plot(x,y,label=lb[i]+' acc:'+str(acc),color=color[i])
plt.xlabel('epsilon (1e-3)')
plt.ylabel('loss')
plt.legend()
plt.title('SGDM sharpness (batchsize 128)')
plt.show()


# prefix = 'sharpness_measure_lb_'
# postfix = '.json'
# fns = ['adam_0.01','lamb_0.02','novograd_0.2','radam_0.0033','rmsprop_0.0003','sgd_0.001','sgdwm_0.001']
# lb = ['Adam max-lr-0.01','LAMB max-lr-0.02','NovoGrad max-lr-0.2','RAdam max-lr-0.0033','RMSProp max-lr-0.0003','SGD max-lr-0.001','SGDM max-lr-0.001']
# x = [5e-4,2e-4,1e-4,5e-5,1e-5,1e-6,0,-1e-6,-1e-5,-5e-5,-1e-4,-2e-4,-5e-4]
# x = [1000*i for i in x]
# x = x[2:-2]
# for i,fn in enumerate(fns):
#     dif,eps,original,loss,acc = json.load(open(prefix+fn+postfix))
#     y = original[:6]+[loss]+original[6:]
#     y[10],y[11] = y[11],y[10]
#     y = y[2:-2]
#     plt.plot(x,y,label=lb[i]+' acc:'+str(acc))
# plt.xlabel('epsilon (1e-3)')
# plt.ylabel('loss')
# plt.legend()
# plt.title('sharpness (batchsize 8192)')
# plt.show()
#
# prefix = 'sharpness_measure_lb_'
# postfix = '.json'
# fns = ['adam_0.01','lamb_0.02','novograd_0.2','radam_0.0033','rmsprop_0.0003']
# x = [5e-4,2e-4,1e-4,5e-5,1e-5,1e-6,0,-1e-6,-1e-5,-5e-5,-1e-4,-2e-4,-5e-4]
# x = [1000*i for i in x]
# x = x[1:]
# for i,fn in enumerate(fns):
#     dif,eps,original,loss,acc = json.load(open(prefix+fn+postfix))
#     y = original[:6]+[loss]+original[6:]
#     y[10],y[11] = y[11],y[10]
#     y = y[1:]
#     plt.plot(x,y,label=lb[i]+' acc:'+str(acc))
# plt.xlabel('epsilon (1e-3)')
# plt.ylabel('loss')
# plt.legend()
# plt.title('sharpness (batchsize 8192)')
# plt.show()

# prefix = 'sharpness_measure_'
# postfix = '.json'
# fns = ['sgdwm_0.2','adam_0.0006','lamb_0.01','lars_2.3','novograd_0.05','radam_0.001']
# lb = ['SGDM max-lr-0.2','Adam max-lr-0.0006','LAMB max-lr-0.01','LARS max-lr-2.3','NovoGrad max-lr-0.05','RAdam max-lr-0.001']
# x = [5e-4,2e-4,1e-4,5e-5,1e-5,1e-6,0,-1e-6,-1e-5,-5e-5,-2e-4,-1e-4,-5e-4]
# x = [1000*i for i in x]
# x[10],x[11]=x[11],x[10]
# x = x[1:-1]
# color = ['r','g','b']
# for i,fn in enumerate(fns):
#     dif,eps,original,loss,acc = json.load(open(prefix+fn+postfix))
#     y = original[:6]+[loss]+original[6:]
#     y[10],y[11] = y[11],y[10]
#     y = y[1:-1]
#     plt.plot(x,y,label=lb[i]+' acc:'+str(acc))
# plt.xlabel('epsilon (1e-3)')
# plt.ylabel('loss')
# plt.legend()
# plt.title('sharpness (batchsize 128)')
# plt.show()
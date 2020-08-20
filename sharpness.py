import torch
import reverse_sgd
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def cal_sharpness(model,data_loader,criterion,epsilon_list=[1e-5]):
    #consider add one more, with eps(1+max(x)) as denominator
    res,res_eps,res_original = [],[],[]
    params = model.state_dict()
    deep_copy_params = {k:params[k] for k in params}
    for eps in epsilon_list:
        currentloss = 0
        print('calculating original loss')
        model.eval()
        total = 0
        correct = 0
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            loss = criterion(outputs, targets)
            currentloss += loss.item()
        currentloss /= len(data_loader)
        print(currentloss)
        print('acc:' + str(correct / total))
        optimizer = reverse_sgd.REV_SGD(model.parameters(),epsilon=eps)
        sharploss = 0
        print('calculating derivatives')
        model.train()
        optimizer.zero_grad()
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print('calculating loss')
        model.eval()
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            sharploss += loss.item()
        sharploss/=len(data_loader)
        print(sharploss)
        res.append((sharploss-currentloss)/(1+currentloss))
        res_eps.append((sharploss-currentloss)/(1+currentloss)/eps)
        res_original.append(sharploss)
        model.load_state_dict(deep_copy_params)
    return res,res_eps,res_original


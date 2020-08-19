import torch
import reverse_sgd
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def cal_sharpness(model,data_loader,criterion,epsilon_list=[1e-5]):
    #consider add one more, with eps(1+max(x)) as denominator
    res,res_eps,res_original = [],[],[]
    params = model.state_dict()
    currentloss = 0
    print('calculating original loss')
    model.eval()
    for batch_idx, (inputs, targets) in enumerate(data_loader):
        print(batch_idx)
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        currentloss += loss.item()
    currentloss/=len(data_loader)
    print(currentloss)
    for eps in epsilon_list:
        optimizer = reverse_sgd.REV_SGD(model.parameters(),epsilon=eps)
        optimizer.zero_grad()
        sharploss = 0
        print('calculating derivatives')
        model.train()
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            print(batch_idx)
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
        optimizer.step()
        print('calculating loss')
        model.eval()
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            print(batch_idx)
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            sharploss += loss.item()
        sharploss/=len(data_loader)
        print(sharploss)
        res.append((sharploss-currentloss)/(1+currentloss))
        res_eps.append((sharploss-currentloss)/(1+currentloss)/eps)
        res_original.append(sharploss)
        model.load_state_dict(params)
    return res,res_eps,res_original


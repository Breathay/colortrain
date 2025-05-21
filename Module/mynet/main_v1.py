# -*- coding: utf-8 -*-
import sys 
homepath = "../../"
sys.path.append(homepath)
from tools.MileServer import MileServer
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import time
from pretrain import Forward, Standard

def InvStandard(s, mean, sd):
    s = s * sd + mean
    return s

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

#device = torch.device("cpu")

def dataLoader(BATCH_SIZE=100):

    data = np.array(pd.read_csv(homepath + "data/dataset/data_w.csv").iloc[:, 1:])

    X = data[:, :3]
    s = data[:, 6:]
    
    standard = np.loadtxt(homepath + "data/standard/standard.txt")
    mean_X, sd_X = standard[:, 2], standard[:, 3]
    
    X = Standard(X, mean_X, sd_X)
    
    s = Standard(s)
    
    X_train, X_test, s_train, s_test = train_test_split(X, s, test_size=0.2,
                                                        random_state=10)
    X_train, X_test = torch.Tensor(X_train), torch.Tensor(X_test)
    s_train, s_test = torch.Tensor(s_train), torch.Tensor(s_test)
    
    train_dataset = torch.utils.data.TensorDataset(X_train, s_train)
    test_dataset = torch.utils.data.TensorDataset(X_test, s_test)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, 
                                               batch_size=BATCH_SIZE,
                                               shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=BATCH_SIZE,
                                              shuffle=False)
    return train_loader, test_loader

class Backward(nn.Module):
    def __init__(self, n_gauss=25):
        super(Backward, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(3, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 200),
            nn.ReLU(inplace=True),
            )
        
        self.param = nn.ModuleList([nn.Linear(200, 4) for i in range(3 * n_gauss)])
        self.n_gauss = n_gauss
    
    def reparametrization(self, cache, weight):
        rand = torch.randn(cache[0][0].size()).to(device)
        
        samples_index = torch.zeros(rand.size()).T
        for i in range(4):
            samples_index[i] = torch.multinomial(weight[:, :, i].T, 1, replacement=True).squeeze()
        samples_index = samples_index.T
        
        x = torch.zeros(cache[0][0].size()).to(device)
        for i in range(rand.shape[0]):
            for j in range(rand.shape[1]):
                index = samples_index[i, j]
                miu, sigma = cache[int(index.numpy())][0], cache[int(index.numpy())][1]
                x[i, j] = rand[i, j] * sigma[i, j] + miu[i, j]
        return x
    
    # def reparametrization(self, cache, weights):
        
    #     rand = torch.randn(cache[0][0].size()).to(device)
        
    #     weights = weights.reshape(-1, 4, self.n_gauss)

    #     samples_index = torch.zeros(rand.size()).T
    #     for i in range(4):
    #         samples_index[i] = torch.multinomial(weights[:, i, :], 1, replacement=True).squeeze()
    #     samples_index = samples_index.T
        
    #     x = torch.zeros(cache[0][0].size()).to(device)
        
    #     for i in range(rand.shape[0]):
    #         for j in range(rand.shape[1]):
    #             index = samples_index[i, j]
    #             miu, sigma = cache[int(index.numpy())][0], cache[int(index.numpy())][1]
    #             x[i, j] = rand[i, j] * sigma[i, j] + miu[i, j]

    #     return rand
    
    def forward(self, x0, pretrain=None):
        x = self.mlp(x0)
        cache = []
        weight = torch.zeros(self.n_gauss, x.shape[0], 4)
        for i in range(0, len(self.param), 3):
            miu = self.param[i](x)
            sigma = torch.abs(self.param[i + 1](x))
            pai = torch.abs(self.param[i + 2](x))
            cache.append((miu, sigma, pai))
            weight[int(i / 3)] = pai
            
        out = self.reparametrization(cache, weight) 
        
        if pretrain is not None:
            Xpred = pretrain(out)
            loss = F.mse_loss(x0, Xpred).item()
            for i in range(10):
                temp_out = self.reparametrization(cache, weight) 
                temp_Xpred = pretrain(temp_out)
                temp_loss = F.mse_loss(temp_Xpred, x0).item()
                if temp_loss < loss:
                    out = temp_out
                    loss = temp_loss
        return out
    
def adjust_learning_rate(optimizer, epoch, lr):
    lr *= (0.6 ** (epoch // 70))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        
if __name__ == "__main__":        

    forward_state = torch.load("forward.pth", map_location=device)
    forward = Forward().to(device)
    forward.load_state_dict(forward_state)
    
    train_loader, test_loader = dataLoader()
    
    backward = Backward().to(device)
    optimizer = optim.Adam(backward.parameters(), lr=1e-3)
    
    start = time.time()
    
    epochs = 100
    loss_list = []
    lr_init = optimizer.param_groups[0]['lr']
    for epoch in range(epochs):
        
        backward.eval()
        torch.save(backward.state_dict(), "backward.pth")
        test_loss = 0
        step = 0
        for i, [Xtest, stest] in enumerate(test_loader):
            Xtest, stest = Xtest.to(device), stest.to(device)
            s_pred = backward(Xtest)
            
            #mean, sd = torch.mean(stest), torch.sd(stest)

            X_pred = forward(s_pred)
            test_loss = test_loss + F.mse_loss(X_pred, Xtest).item()
            step = step + 1
        test_loss = test_loss / step
        loss_list.append(test_loss)
        print("epoch: {}, loss: {:5f}".format(epoch, test_loss))
        print(optimizer.param_groups[0]['lr'])
        
        backward.train(True)
        adjust_learning_rate(optimizer, epoch, lr_init)
        for i, [Xtrain, strain] in enumerate(train_loader):
            Xtrain, strain = Xtrain.to(device), strain.to(device)
            spred = backward(Xtrain, forward)
            Xpred = forward(spred)
            
            loss = F.mse_loss(Xpred, Xtrain)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i % 100 == 0:
                print("step: {}, iter: {}/{}, loss: {:5f}".format(i, \
              i * len(Xtrain), len(train_loader.dataset), loss.item()))
        
    
    loss_list = np.array(loss_list)
    np.savetxt("loss_backward.txt", loss_list)
    
    end = time.time()
    print("Total time : {}".format(end - start))
    
    server = MileServer()
    server.setEmileContent('Train Finished', str(end - start) + ' seconds used')
    server.sendEmile()








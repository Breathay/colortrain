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
from tools.DataLoader import getLoader

def InvStandard(s, mean, sd):
    s = s * sd + mean
    return s

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


class MDN(nn.Module):
    def __init__(self, n_gauss=10):
        super(MDN, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 200),
            nn.ReLU(inplace=True),
        )

        self.pi_net = nn.Linear(200, n_gauss)
        self.mu_net = nn.Linear(200, n_gauss * 4)
        self.sd_net = nn.Linear(200, n_gauss * 4)
        
        self.softmax = nn.Softmax(dim=1)
        
        self.n_gauss = n_gauss

    def forward(self, x0):
        x = self.mlp(x0)
        
        pi = self.softmax(self.pi_net(x))
        mu = self.mu_net(x).reshape(-1, self.n_gauss, 4)
        sigma = torch.exp(self.sd_net(x)).reshape(-1, self.n_gauss, 4)
        
        return pi, mu, sigma
    
# mdn损失函数
def mdn_loss_fn(y, mu, sigma, pi):
    
    y = y.reshape(y.shape[0], 1, y.shape[1])
    
    m = torch.distributions.Normal(loc=mu, scale=sigma)
    loss = torch.exp(m.log_prob(y))
    loss = torch.sum(loss * pi.reshape(pi.shape[0], pi.shape[1], 1), dim=1)
    loss = -torch.log(loss)
    return torch.mean(loss)



def adjust_learning_rate(optimizer, epoch, lr):
    lr *= (0.7 ** (epoch // 70))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        
if __name__ == "__main__":        

    train_loader, test_loader = dataLoader()
    
    mdn = MDN().to(device)
    optimizer = optim.Adam(mdn.parameters(), lr=1e-3)
    
    start = time.time()
    
    epochs = 200
    loss_list = []
    lr_init = optimizer.param_groups[0]['lr']
    for epoch in range(epochs):
        
        mdn.eval()
        torch.save(mdn.state_dict(), "model.pth")
        test_loss = 0
        step = 0
        for i, [Xtest, stest] in enumerate(test_loader):
            Xtest, stest = Xtest.to(device), stest.to(device)
            pi, mu, sigma = mdn(Xtest)
            loss = mdn_loss_fn(stest, mu, sigma, pi)
            test_loss = test_loss + loss.item()
            step = step + 1
        test_loss = test_loss / step
        loss_list.append(test_loss)
        print("epoch: {}, loss: {:5f}".format(epoch, test_loss))
        print(optimizer.param_groups[0]['lr'])
        
        mdn.train(True)
        adjust_learning_rate(optimizer, epoch, lr_init)
        for i, [Xtrain, strain] in enumerate(train_loader):
            Xtrain, strain = Xtrain.to(device), strain.to(device)
            pi, mu, sigma = mdn(Xtrain)
            loss = mdn_loss_fn(strain, mu, sigma, pi)
            
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








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
from Module.forward_xyz.pretrain import Forward
import torch.distributions as dist
from tools.DataLoader import getLoader

def InvStandard(s, mean, sd):
    s = s * sd + mean
    return s

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

class Backward(nn.Module):
    def __init__(self, n_gauss=25):
        super(Backward, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 200),
            nn.ReLU(inplace=True),
            nn.Linear(200, 4),
            )
    
    def forward(self, x0):
        x = self.mlp(x0)
        return x
    
def adjust_learning_rate(optimizer, epoch, lr):
    lr *= (0.7 ** (epoch // 70))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        
if __name__ == "__main__":        

    forward_state = torch.load(homepath + "Module/forward_xyz/forward_xy.pth", map_location=device)
    forward = Forward().to(device)
    forward.load_state_dict(forward_state)
    
    train_loader, test_loader, mean_s, sd_s = getLoader("data_w_new.csv")
    
    backward = Backward().to(device)
    optimizer = optim.Adam(backward.parameters(), lr=1e-3)
    
    start = time.time()
    
    epochs = 100
    loss_list = []
    lr_init = optimizer.param_groups[0]['lr']
    for epoch in range(epochs):
        
        backward.eval()
        torch.save(backward.state_dict(), "model.pth")
        test_loss = 0
        step = 0
        for i, [Xtest, stest] in enumerate(test_loader):
            Xtest, stest = Xtest.to(device), stest.to(device)
            s_pred = backward(Xtest)

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
            spred = backward(Xtrain)
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








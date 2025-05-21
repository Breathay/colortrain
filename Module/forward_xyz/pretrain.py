# -*- coding: utf-8 -*-
import sys 
homepath = "../../"
sys.path.append(homepath)
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import time
from tools.DataLoader import getLoader

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

class Forward(nn.Module):
    def __init__(self, output_dim=2):
        super(Forward, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(4, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, output_dim)
            )
    def forward(self, x):
        out = self.mlp(x)
        return out
    
def adjust_learning_rate(optimizer, epoch, lr):
    lr *= (0.8 ** (epoch // 60))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        
if __name__ == "__main__":
    
    train_loader, test_loader, mean_s, sd_s = getLoader()
    forward = Forward().to(device)
    optimizer = optim.Adam(forward.parameters(), lr=1e-3)
    
    start = time.time()        
    
    epochs = 200
    loss_list = []
    lr_init = optimizer.param_groups[0]['lr']
    for epoch in range(epochs):
        forward.eval()
        torch.save(forward.state_dict(), "forward_xy.pth")
        test_loss = 0
        step = 0
        for i, [Xtest, stest] in enumerate(test_loader):
            Xtest, stest = Xtest.to(device), stest.to(device)
            X_pred = forward(stest)
            test_loss += F.mse_loss(X_pred, Xtest).item()
            step += 1
        test_loss = test_loss / step
        loss_list.append(test_loss)
        print("epoch: {}, loss: {:5f}".format(epoch, test_loss))
        print(optimizer.param_groups[0]['lr'])
        
        forward.train(True)
        adjust_learning_rate(optimizer, epoch, lr_init)
        for i, (Xtrain, strain) in enumerate(train_loader):
            Xtrain, strain = Xtrain.to(device), strain.to(device)
            Xpred = forward(strain)
            loss = F.mse_loss(Xpred, Xtrain)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i % 100 == 0:
                print("step: {}, iter: {}/{}, loss: {:5f}".format(i, \
              i * len(Xtrain), len(train_loader.dataset), loss.item()))
                    
    loss_list = np.array(loss_list)
    np.savetxt("loss_forward_xy.txt", loss_list)
    
    end = time.time()
    print("Total time : {}".format(end - start))
    
    # from MileServer import MileServer
    # server = MileServer()
    # server.setEmileContent('Train Finished', str(end - start) + ' seconds used')
    # server.sendEmile()


        
        
        
        

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        

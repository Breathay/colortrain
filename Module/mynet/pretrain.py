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

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

def Standard(s, mean=None, sd=None):
    if mean is None:
        mean = np.mean(s, axis=0)
    if sd is None:
        sd = np.std(s, axis=0)
    s = (s - mean) / sd
    return s


def InvStandard(s, mean, sd):
    s = s * sd + mean
    return s

def getLoader(BATCH_SIZE=100):
    data = np.array(pd.read_csv(homepath + "data/dataset/data_w.csv").iloc[:, 1:])

    X, s = data[:, :3], data[:, 6:]
    
    standard = np.loadtxt(homepath + "data/standard/standard.txt")
    mean_X, sd_X = standard[:, 2], standard[:, 3]
    
    X = Standard(X, mean_X, sd_X)
    s = Standard(s)
    X_train, X_test, s_train, s_test = train_test_split(X, s, test_size=0.2,
                                                        random_state = 42)
    X_train, s_train = torch.Tensor(X_train), torch.Tensor(s_train)
    X_test, s_test = torch.Tensor(X_test), torch.Tensor(s_test)
    
    train_dataset = torch.utils.data.TensorDataset(X_train, s_train)
    test_dataset = torch.utils.data.TensorDataset(X_test, s_test)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=BATCH_SIZE,
                                               shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=BATCH_SIZE,
                                              shuffle=False)
    return train_loader, test_loader

class Forward(nn.Module):
    def __init__(self,):
        super(Forward, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(4, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 3)
            )
    def forward(self, x):
        out = self.mlp(x)
        return out
    
def adjust_learning_rate(optimizer, epoch, lr):
    lr *= (0.6 ** (epoch // 60))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        
if __name__ == "__main__":
    
    train_loader, test_loader = getLoader()
    forward = Forward().to(device)
    optimizer = optim.Adam(forward.parameters(), lr=1e-3)
    
    start = time.time()        
    
    epochs = 300
    loss_list = []
    lr_init = optimizer.param_groups[0]['lr']
    for epoch in range(epochs):
        forward.eval()
        torch.save(forward.state_dict(), "forward.pth")
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
    np.savetxt("loss_forward.txt", loss_list)
    
    end = time.time()
    print("Total time : {}".format(end - start))
    
    # from MileServer import MileServer
    # server = MileServer()
    # server.setEmileContent('Train Finished', str(end - start) + ' seconds used')
    # server.sendEmile()


        
        
        
        

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        

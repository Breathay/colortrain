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
import torch.distributions as dist

def InvStandard(s, mean, sd):
    s = s * sd + mean
    return s

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

#device = torch.device("cpu")

def dataLoader(BATCH_SIZE=100):

    data = np.array(pd.read_csv(homepath + "data/dataset/data_old.csv").iloc[:, 1:])

    X = data[:, :3]
    s = data[:, 3:]
    
    # standard = np.loadtxt(homepath + "data/standard/standard.txt")
    # mean_X, sd_X = standard[:, 2], standard[:, 3]
    
    
    X = Standard(X)
    
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
        
        self.pi_net = nn.Linear(200, n_gauss)
        self.mu_net = nn.Linear(200, n_gauss * 4)
        self.sd_net = nn.Linear(200, n_gauss * 4)
        
        self.softmax = nn.Softmax(dim=1)
        
        self.n_gauss = n_gauss
    
    def reparametrization(self, mu, pi, sd, n_samples):
        batch_size, num_gaussians, output_dim = mu.shape
        
        indices = torch.multinomial(pi, n_samples, replacement=True)  
        indices = indices.unsqueeze(-1).expand(-1, -1, output_dim)
        
        selected_mu = torch.gather(mu, 1, indices)  
        selected_sd = torch.gather(sd, 1, indices)  
        
        normal_dist = dist.Normal(selected_mu, selected_sd)
        samples = normal_dist.rsample()  # 使用 rsample 支持反向传播
        
        return samples

    def forward(self, x0, pretrain=None, n_samples=10):
        x = self.mlp(x0)
        
        pi = self.softmax(self.pi_net(x))
        mu = self.mu_net(x).reshape(-1, self.n_gauss, 4)
        sigma = torch.exp(self.sd_net(x)).reshape(-1, self.n_gauss, 4)
        
        out = self.reparametrization(mu, pi, sigma, n_samples)
        final_out = out[:, 0, :]
        
        if pretrain is not None:
            Xpred = pretrain(final_out)
            loss = F.mse_loss(x0, Xpred).item()
            
            for i in range(1, n_samples):

                temp_out = out[:, i, :]
                temp_Xpred = pretrain(temp_out)
                temp_loss = F.mse_loss(temp_Xpred, x0).item()
                if temp_loss < loss:
                    final_out = temp_out
                    loss = temp_loss
        return final_out
        
    
def adjust_learning_rate(optimizer, epoch, lr):
    lr *= (0.6 ** (epoch // 30))
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








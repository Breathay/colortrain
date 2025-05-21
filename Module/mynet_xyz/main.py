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

# device = torch.device("cpu")

class Backward(nn.Module):
    def __init__(self, n_gauss=10, input_dim=2, output_dim=4, mean_s=None, sd_s=None):
        super(Backward, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            )
        
        self.pi_net = nn.Linear(128, n_gauss)
        self.mu_net = nn.Linear(128, n_gauss * output_dim)
        self.sd_net = nn.Linear(128, n_gauss * output_dim)
        
        self.softmax = nn.Softmax(dim=1)
        
        self.n_gauss = n_gauss
        self.output_dim = output_dim
        self.mean_s = mean_s
        self.sd_s = sd_s
        
    def reparametrization(self, mu, pi, sd, n_samples):
        batch_size, num_gaussians, output_dim = mu.shape
        
        indices = torch.multinomial(pi, n_samples, replacement=True)  
        indices = indices.unsqueeze(-1).expand(-1, -1, output_dim)
        
        selected_mu = torch.gather(mu, 1, indices)  
        selected_sd = torch.gather(sd, 1, indices)  
        
        normal_dist = dist.Normal(selected_mu, selected_sd)
        samples = normal_dist.rsample()  # 使用 rsample 支持反向传播
        
        return samples
    
    def getWrongIndex(self, s_pred, mean_s, sd_s):
        s = s_pred.detach().cpu().numpy()
        s = InvStandard(s, mean_s, sd_s)
        # 获取s[i,j]小于20的索引
        
        index1 = np.where((s < 20).any(2))
        # 获取和大于1000的索引
        index2 = np.where(s.sum(axis=2) > 1000)
        # 获取大于500的索引
        index3 = np.where((s > 500).any(2))
        # 合并索引
        index = (np.concatenate((index1[0], index2[0], index3[0])), np.concatenate((index1[1], index2[1], index3[1])))
        return index
    
    def forward(self, x0, pretrain=None, n_samples=20):
        x = self.mlp(x0)
        
        pi = self.softmax(self.pi_net(x))
        mu = self.mu_net(x).reshape(-1, self.n_gauss, self.output_dim)
        sigma = torch.exp(self.sd_net(x)).reshape(-1, self.n_gauss, self.output_dim)
        
        out = self.reparametrization(mu, pi, sigma, n_samples)
        final_out = out[:, 0, :]
        
        if pretrain is not None:
            # 一次性计算所有样本的预测值
            batch_size, n_samples, output_dim = out.shape
            out_flat = out.view(-1, output_dim)  # (batch_size*n_samples, output_dim)
            pred_flat = pretrain(out_flat)  # (batch_size*n_samples, input_dim)
            pred = pred_flat.view(batch_size, n_samples, -1)  # (batch_size, n_samples, input_dim)
            
            # 计算每个样本的误差（逐点计算）
            x0_expanded = x0.unsqueeze(1).expand_as(pred)  # (batch_size, n_samples, input_dim)
            errors = F.mse_loss(pred, x0_expanded, reduction='none')  # (batch_size, n_samples, input_dim)
            errors = errors.mean(dim=-1)  # (batch_size, n_samples)

            index = self.getWrongIndex(out, self.mean_s, self.sd_s)
            errors[index] = 1e10
            
            # 对每个数据点选择误差最小的样本
            min_indices = torch.argmin(errors, dim=1)  # (batch_size,)
            final_out = out[torch.arange(batch_size), min_indices, :]  # (batch_size, output_dim)
        return final_out
    
def adjust_learning_rate(optimizer, epoch, lr):
    lr *= (0.7 ** (epoch // 70))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        
if __name__ == "__main__":        

    forward_state = torch.load(homepath + "Module/forward_xyz/forward_xy.pth", map_location=device)
    forward = Forward().to(device)
    forward.load_state_dict(forward_state)
    
    train_loader, test_loader, mean_s, sd_s = getLoader("data_w_new.csv", 256)
    
    backward = Backward(mean_s=mean_s, sd_s=sd_s).to(device)
    optimizer = optim.Adam(backward.parameters(), lr=1e-3)
    
    start = time.time()
    
    epochs = 200
    loss_list = []
    lr_init = optimizer.param_groups[0]['lr']
    for epoch in range(epochs):
        
        backward.eval()
        torch.save(backward.state_dict(), "model.pth")
        test_loss = 0
        step = 0
        for i, [Xtest, stest] in enumerate(test_loader):
            Xtest, stest = Xtest.to(device), stest.to(device)
            s_pred = backward(Xtest, forward)
            
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








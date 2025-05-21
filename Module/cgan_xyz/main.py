# -*- coding: utf-8 -*-
import sys
import time 
homepath = "../../"
sys.path.append(homepath)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from Module.forward_xyz.pretrain import Forward
from tools.DataLoader import getLoader

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

class Generator(nn.Module):
    def __init__(self, input_size=2, output_size=4, noise_dim=10, hidden_dim=64):
        super(Generator, self).__init__()

        self.input_size = input_size

        self.net = nn.Sequential(*[nn.Linear(input_size + noise_dim, hidden_dim),
                                   nn.LeakyReLU(0.2),
                                   nn.Linear(hidden_dim, hidden_dim),
                                   nn.LeakyReLU(0.2),
                                   nn.Linear(hidden_dim, hidden_dim),
                                   nn.LeakyReLU(0.2),
                                   nn.Linear(hidden_dim, output_size)])

    def forward(self, x, noise):
        y = self.net(torch.cat((x, noise), dim=1))
        return y

class Discriminator(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim=64):
        super(Discriminator, self).__init__()

        self.net = nn.Sequential(*[nn.Linear(input_size+output_size, hidden_dim),
                                   nn.LeakyReLU(0.2),
                                   #nn.BatchNorm1d(hidden_dim), #
                                   #don't use batch norm for the D input layer and G output layer to aviod the oscillation and model instability 
                                   nn.Linear(hidden_dim, hidden_dim),
                                   nn.LeakyReLU(0.2),
                                   nn.Linear(hidden_dim, hidden_dim),
                                   nn.LeakyReLU(0.2),
                                   nn.Linear(hidden_dim, hidden_dim),
                                   nn.LeakyReLU(0.2)])
        

        # 输出层
        self.adv_layer = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())
        # self.aux_layer = nn.Sequential(nn.Linear(128, 3))

    def forward(self, y_fake, x):
        h = self.net(torch.cat((y_fake, x), dim=1))
        validity = self.adv_layer(h)
        # label = self.aux_layer(h)

        return validity


class cGAN(nn.Module):
    def __init__(self, input_size, output_size, noise_dim=3, hidden_dim=64):
        super(cGAN, self).__init__()

        self.generator = Generator(
            input_size, output_size, noise_dim=noise_dim, hidden_dim=hidden_dim)
        self.discriminator = Discriminator(
            output_size, input_size, hidden_dim=hidden_dim)

        self.noise_dim = noise_dim

    def forward(self, x, noise):

        y_fake = self.generator(x, noise)
        validity = self.discriminator(y_fake, x)

        return validity

    def sample_noise(self, batch_size, prior=1):

        if prior == 1:
            z = torch.tensor(np.random.normal(0, 1, (batch_size, self.noise_dim))).float()
        else:
            z = torch.tensor(np.random.uniform(0, 1, (batch_size, self.noise_dim))).float()
        return z

    def sample_noise_M(self, batch_size):
        M = 100
        z = torch.tensor(np.random.normal(
            0, 1, (batch_size*M, self.noise_dim))).float()
        return z
    
def train(model, train_loader, optimizer_G, optimizer_D, criterion, prior):

    model.train()
    g_loss_epoch = 0
    d_loss_epoch = 0

    for i, (x, y) in enumerate(train_loader):

        batch_size = len(x)
        x, y = x.to(device), y.to(device)

        # 真实值

        valid = torch.ones(batch_size, 1).to(device)
        fake = torch.zeros(batch_size, 1).to(device)

        # 训练生成器

        optimizer_G.zero_grad()
        z = model.sample_noise(batch_size, prior).to(device)

        gen_y = model.generator(x, z)
        validity = model.discriminator(gen_y, x)
        g_loss = criterion(validity, valid)
        g_loss.backward()
        optimizer_G.step()

        # 训练判别器

        optimizer_D.zero_grad()
        # 在真实数据上

        real_pred = model.discriminator(y, x)
        d_loss_real = criterion(real_pred, valid)

        # 在生成数据上
        gen_y = model.generator(x, z)
        fake_pred = model.discriminator(gen_y, x)
        d_loss_fake = criterion(fake_pred, fake)

        d_loss = (d_loss_real + d_loss_fake)/2
        d_loss.backward()
        optimizer_D.step()

        g_loss_epoch += g_loss * batch_size
        d_loss_epoch += d_loss * batch_size

        if i % 100 == 0:
            print("step: {}, iter: {}/{}, g_loss: {:5f}, d_loss: {:5f}".format(i, i * len(x), 
            len(train_loader.dataset), g_loss.item(), d_loss.item()))

    g_loss_epoch, d_loss_epoch = g_loss_epoch / len(train_loader.dataset), d_loss_epoch / len(train_loader.dataset)

    return g_loss_epoch + d_loss_epoch, g_loss_epoch, d_loss_epoch

def evaluate(model, test_loader, forward_model, criterion, prior):

    mse_criterion = nn.MSELoss()

    model.eval()

    torch.save(model.state_dict(), "model.pth")
    
    dataloader = test_loader
    with torch.no_grad():
        g_loss_epoch = 0
        d_loss_epoch = 0
        error_epoch = 0
        # x: 颜色, y: 结构
        idx = 0
        for x, y in dataloader:
            batch_size = len(x)
            x, y = x.to(device), y.to(device)

            valid = torch.ones(batch_size, 1).to(device)
            fake = torch.zeros(batch_size, 1).to(device)

            # 生成器的误差 

            z = model.sample_noise(batch_size, prior).to(device)
            gen_y = model.generator(x, z)

            
            validity = model.discriminator(gen_y, x)
            g_loss = criterion(validity, valid)
            
            # 判别器在真实数据上的误差 
            
            real_pred = model.discriminator(y, x)
            d_loss_real = criterion(real_pred, valid)
            
            # 判别器在生成数据上的误差 

            fake_pred = model.discriminator(gen_y, x)
            d_loss_fake = criterion(fake_pred, fake)
            d_loss = (d_loss_real + d_loss_fake)/2

            g_loss_epoch += g_loss#  * batch_size
            d_loss_epoch += d_loss#  * batch_size
            # 正向模型计算误差
            X_pred = forward_model(gen_y)

            error_epoch += mse_criterion(X_pred, x)
            idx += 1
        
        g_loss_epoch = g_loss_epoch / idx
        d_loss_epoch = d_loss_epoch / idx
        error_epoch = error_epoch / idx
        
    return g_loss_epoch, d_loss_epoch, error_epoch

# 使用示例
if __name__ == "__main__":

    input_dim = 2
    output_dim = 4
    noise_dim = 10
    lr = 0.001
    prior = 1
    epoch_lr_de = 70
    lr_de = 0.8

    # 数据集
    train_loader, test_loader, mean_s, sd_s = getLoader()

    model = cGAN(input_dim, output_dim, noise_dim).to(device)

    optimizer_G = optim.Adam(model.generator.parameters(), lr=lr)
    optimizer_D = optim.Adam(model.discriminator.parameters(), lr=lr)

    scheduler_G = StepLR(optimizer_G, step_size=epoch_lr_de, gamma=lr_de)
    scheduler_D = StepLR(optimizer_D, step_size=epoch_lr_de, gamma=lr_de)
    
    criterion = nn.BCELoss()

    forward_state = torch.load(homepath + "Module/forward_xyz/forward_xy.pth", map_location=device)
    forward_model = Forward(2).to(device)
    forward_model.load_state_dict(forward_state)
    
    start = time.time()        
    
    epochs = 300
    loss_list = []

    for epoch in range(epochs):

        loss_train, loss_train_g, loss_train_d = train(model, train_loader, optimizer_G, optimizer_D, criterion, prior)
        loss_test_g, loss_test_d, error_test = evaluate(model, test_loader, forward_model, criterion, prior)

        print("epoch: {}, g_loss: {:5f}, d_loss: {:5f}, error: {:5f}".format(epoch, loss_test_g, loss_test_d, error_test))
        print("-" * 10)

        scheduler_D.step()
        scheduler_G.step()

        loss_list.append(error_test.item())

    loss_list = np.array(loss_list)
    np.savetxt("loss_backward.txt", loss_list)

        




        
        
        
        
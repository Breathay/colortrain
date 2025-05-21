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

class cVAE_GSNN1(nn.Module):
    # a deeper but narrow network
    def __init__(self, input_size, latent_dim, hidden_dim=64, forward_dim=2):
        super(cVAE_GSNN1, self).__init__()

        self.input_size = input_size
        self.latent_dim = latent_dim

        # encoder
        self.encoder = nn.Sequential(*[nn.Linear(input_size+forward_dim, hidden_dim),
                                       nn.ReLU(),
                                       # nn.BatchNorm1d(hidden_dim),
                                       nn.Linear(hidden_dim, hidden_dim),
                                       nn.ReLU(),
                                       # nn.BatchNorm1d(hidden_dim),
                                       nn.Linear(hidden_dim, hidden_dim),
                                       nn.ReLU(),
                                       # nn.BatchNorm1d(hidden_dim)
                                       ])


        self.forward_net = nn.Sequential(*[nn.Linear(forward_dim, hidden_dim),
                                   nn.ReLU(),
                                   # nn.BatchNorm1d(hidden_dim),
                                   nn.Linear(hidden_dim, hidden_dim),
                                   nn.ReLU(),
                                   # nn.BatchNorm1d(hidden_dim),
                                   nn.Linear(hidden_dim, hidden_dim),
                                   nn.ReLU(),
                                   nn.Linear(hidden_dim, input_size)])
        
        self.mu_head = nn.Linear(hidden_dim, latent_dim)
        self.logvar_head = nn.Linear(hidden_dim, latent_dim)

        # decoder
        self.decoder = nn.Sequential(*[nn.Linear(latent_dim + forward_dim, hidden_dim),
                                       nn.ReLU(),
                                       # nn.BatchNorm1d(hidden_dim),
                                       nn.Linear(hidden_dim, hidden_dim),
                                       nn.ReLU(),
                                       # nn.BatchNorm1d(hidden_dim),
                                       nn.Linear(hidden_dim, hidden_dim),
                                       nn.ReLU(),
                                       # nn.BatchNorm1d(hidden_dim),
                                       nn.Linear(hidden_dim, input_size)])

    def encode(self, x):
        h = self.encoder(x)
        return self.mu_head(h), self.logvar_head(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z, y):
        temp = torch.cat((z,y), dim=1)
        recon_x = self.decoder(torch.cat((z, y), dim=1))
        return recon_x

    def forward(self, x, y):
        x_hat = self.forward_net(y)
        h = self.encoder(torch.cat((x, y), dim=1))

        mu, logvar = self.mu_head(h), self.logvar_head(h)
        z = self.reparameterize(mu, logvar)
        
        return self.decode(z, y), mu, logvar, x_hat
    
    def inference(self, y):
        x = self.forward_net(y)
        h = self.encoder(torch.cat((x, y), dim=1))
        mu, logvar = self.mu_head(h), self.logvar_head(h)
        z = self.reparameterize(mu, logvar)

        return self.decode(z, y), mu, logvar, x

class cVAE_hybrid(nn.Module):

    def __init__(self, forward_model, vae_model):
        super(cVAE_hybrid, self).__init__()
        self.forward_model = forward_model
        self.vae_model = vae_model

    def forward(self, x, y):
        # the prediction is based on cVAE_GSNN model
        '''
        Pass the desired target x to the vae_hybrid network.
        '''
        
        x_pred, mu, logvar, x_hat = self.vae_model(x, y)
        
        y_pred = self.forward_model(x_pred)
        return x_pred, mu, logvar, x_hat, y_pred 

    def pred(self, x):
        pred = self.forward_model(x)
        return pred
    
def train(model, train_loader, optimizer, criterion):

    # x: structure ; y: CIE 

    model.vae_model.train()
    model.forward_model.eval()

    loss_epoch = 0

    for i, (y, x) in enumerate(train_loader):
        
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()

        recon_x, mu, logvar, x_hat,  y_pred = model(x, y)

        recon_loss = criterion(recon_x, x)
        replace_loss = criterion(x_hat, x)
        KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        pred_loss = criterion(y_pred, y)

        loss = recon_loss + replace_loss + KLD + pred_loss

        loss.backward()
        optimizer.step()

        loss_epoch += loss * len(x)
        if i % 100 == 0:
            print("step: {}, iter: {}/{}, loss: {:5f}".format(i, i * len(x), len(train_loader.dataset), loss.item()))

    loss_epoch = loss_epoch / len(train_loader.dataset)

    return loss_epoch

def evaluate(model, test_loader, criterion):

    # x: structure ; y: CIE 

    model.eval()

    torch.save(model.vae_model.state_dict(), "model.pth")

    dataloader = test_loader

    with torch.no_grad():
        loss_epoch = 0
        loss_pred = 0

        for y, x in dataloader:

            x, y = x.to(device), y.to(device)

            recon_x, mu, logvar, x_hat,  y_pred = model(x, y)

            recon_loss = criterion(recon_x, x)
            replace_loss = criterion(x_hat, x)
            KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            pred_loss = criterion(y_pred, y)

            loss = recon_loss + replace_loss + KLD
            
            loss_epoch += loss*len(x)
            loss_pred += pred_loss*len(x)
        
        loss_epoch = loss_epoch / len(dataloader.dataset)
        loss_pred = loss_pred / len(dataloader.dataset)

    return loss_epoch, loss_pred

# 使用示例
if __name__ == "__main__":

    input_dim = 4
    output_dim = 2
    latent_dim = 10
    lr = 0.001
    prior = 1
    epoch_lr_de = 100
    lr_de = 0.8

    # 数据集
    train_loader, test_loader, mean_s, sd_s = getLoader()

    forward_state = torch.load(homepath + "Module/forward_xyz/forward_xy.pth", map_location=device)
    forward_model = Forward().to(device)
    forward_model.load_state_dict(forward_state)
    
    vae_model = cVAE_GSNN1(input_dim, latent_dim).to(device)
    model = cVAE_hybrid(forward_model, vae_model)

    # set up optimizer and criterion 

    optimizer = torch.optim.Adam(model.vae_model.parameters(), lr=lr)
    

    scheduler = StepLR(optimizer, step_size=epoch_lr_de, gamma=lr_de)

    criterion = nn.MSELoss()
    
    # start training 
    
    start = time.time()        
    
    epochs = 300
    loss_list = []

    for epoch in range(epochs):

        loss_train = train(model, train_loader, optimizer, criterion)
        loss_val, loss_pred = evaluate(model, test_loader, criterion)
        
        lr = scheduler.get_lr()[0]

        print("epoch: {}, loss_val: {:5f}, loss_pred: {:5f}".format(epoch, loss_val, loss_pred))
        print("-" * 10)

        loss_list.append(loss_pred.item())
    
    loss_list = np.array(loss_list)
    np.savetxt("loss_backward.txt", loss_list)

        




        
        
        
        
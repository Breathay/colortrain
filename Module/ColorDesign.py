# -*- coding: utf-8 -*-
import matlab
import matlab.engine
import sys
homepath = "../"
sys.path.append(homepath)
sys.path.append("E:/程序/python/colortrain_1.18_sih_sio")
from tools.ref2color import getColor
from tools.getCData import getSpectral
import torch
from Module.forward_xyz.pretrain import Forward
from Module.vae_xyz.main import cVAE_GSNN1
from Module.cgan_xyz.main import cGAN
from Module.mynet_xyz.main import Backward as MPSN
from Module.mdn_xyz.main import MDN
from Module.tnn_xyz.main import Backward as TNN
import numpy as np

#计算设备
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

def Standard(X, mean=None, sd=None):
    if mean is None:
        mean = np.mean(X, axis=0)
    if sd is None:
        sd = np.std(X, axis=0)
    X = (X - mean) / sd
    return X

####################################################################################
class DesignColor:
    def __init__(self, mean_X, sd_X, mean_s, sd_s):
        self.mean_X = mean_X
        self.sd_X = sd_X
        self.mean_s = mean_s
        self.sd_s = sd_s
        self.name = None
    
    def setupModel(self,):
        return
    
    def loadModel(self,):
        return
    
    def loadIndex(self, index_SiH, index_SiO):
        self.index_SiH = index_SiH
        self.index_SiO = index_SiO
    
    def InvStandard(self, X, mean, sd):
        if mean is not None and sd is not None:
            X = X * sd + mean
        return X
    
    def predict(self, color, nsamples=1):
        return
    
    def getSpectral(self, s):
        #建立matlab引擎
        engine = matlab.engine.start_matlab()
        engine.addpath(engine.genpath("E:/程序/python/reticolo_allege_v9"))
        engine.addpath(engine.genpath(homepath + '/tools'))
        
        R = getSpectral(s, self.index_SiH, self.index_SiO, engine)
        return R
    
    def getColor(self, R):
        rgb, xyz = getColor(R)
        return rgb, xyz

class MPSNDesign(DesignColor):
    def __init__(self, mean_X, sd_X, mean_s, sd_s):
        super().__init__(mean_X, sd_X, mean_s, sd_s)
        self.name = "mynet"
        

    def setupModel(self,):
        self.forward = Forward().to(device)
        self.backward = MPSN(mean_s=self.mean_s, sd_s=self.sd_s).to(device)

    def loadModel(self, forward_state, backward_state):
        self.forward.load_state_dict(forward_state)
        self.backward.load_state_dict(backward_state)
    
    def predict(self, color, nsamples=10):
        
        #网络输入
        input_color_tensor = torch.Tensor(color).to(device) #转化为Tensor
        #预测结构
        s_pred = self.backward(input_color_tensor, self.forward, nsamples)
        #预测颜色
        X_pred = self.forward(s_pred)
        #转移出计算设备
        s_pred = s_pred.detach().cpu().numpy()
        X_pred = X_pred.detach().cpu().numpy()
        # 逆归一化
        s_pred = self.InvStandard(s_pred, self.mean_s, self.sd_s)
        X_pred = self.InvStandard(X_pred, self.mean_X, self.sd_X)
        return s_pred, X_pred

class MDNDesign(DesignColor):
    def __init__(self, mean_X, sd_X, mean_s, sd_s):
        super().__init__(mean_X, sd_X, mean_s, sd_s)
        self.name = "mdn"
    
    def setupModel(self,):
        self.forward = Forward().to(device)
        self.model = MDN().to(device)
        
    def loadModel(self, forward_state, model_state):
        self.forward.load_state_dict(forward_state)
        self.model.load_state_dict(model_state)
    
    def predict(self, color, nsamples=1):
        #网络输入
        input_color_tensor = torch.Tensor(color).to(device) #转化为Tensor
        
        # 预测结构
        pi, mu, sigma = self.model(input_color_tensor)

        k = torch.multinomial(pi, 1, replacement=True)
        a = torch.normal(mu, sigma)
        s_pred = torch.gather(a, 1, k.unsqueeze(-1).expand(-1, -1, 4))[:, 0, :]

        X_pred = self.forward(s_pred)

        #转移出计算设备
        s_pred = s_pred.detach().cpu().numpy()
        X_pred = X_pred.detach().cpu().numpy()
        # 逆归一化
        s_pred = self.InvStandard(s_pred, self.mean_s, self.sd_s)
        X_pred = self.InvStandard(X_pred, self.mean_X, self.sd_X)
        return s_pred, X_pred

class TNNDesign(DesignColor):
    def __init__(self, mean_X, sd_X, mean_s, sd_s):
        super().__init__(mean_X, sd_X, mean_s, sd_s)
        self.name = "tnn"
        
    def setupModel(self,):
        self.forward = Forward().to(device)
        self.model = TNN().to(device)
    
    def loadModel(self, forward_state, model_state):
        self.forward.load_state_dict(forward_state)
        self.model.load_state_dict(model_state)
    
    def predict(self, color, nsamples=1):
        #网络输入
        input_color_tensor = torch.Tensor(color).to(device) #转化为Tensor
        
        # 预测结构
        s_pred = self.model(input_color_tensor)
        X_pred = self.forward(s_pred)

        #转移出计算设备
        s_pred = s_pred.detach().cpu().numpy()
        X_pred = X_pred.detach().cpu().numpy()
        # 逆归一化
        s_pred = self.InvStandard(s_pred, self.mean_s, self.sd_s)
        X_pred = self.InvStandard(X_pred, self.mean_X, self.sd_X)
        return s_pred, X_pred
        
class CGANDesign(DesignColor):
    def __init__(self, mean_X, sd_X, mean_s, sd_s):
        super().__init__(mean_X, sd_X, mean_s, sd_s)
        self.name = "cgan"
    

    def setupModel(self,):
        self.model = cGAN(input_size=2, output_size=4, noise_dim=10).to(device)
        self.forward = Forward().to(device)
    
    def loadModel(self, forward_state, model_state):
        self.forward.load_state_dict(forward_state)
        self.model.load_state_dict(model_state)
    
    def predict(self, color, nsamples=1):

        #网络输入
        input_color_tensor = torch.Tensor(color).to(device) #转化为Tensor

        # 预测结构
        z = self.model.sample_noise(len(input_color_tensor), 1).to(device)
        s_pred = self.model.generator(input_color_tensor, z)
        X_pred = self.forward(s_pred)

        #转移出计算设备
        s_pred = s_pred.detach().cpu().numpy()
        X_pred = X_pred.detach().cpu().numpy()
        # 逆归一化
        s_pred = self.InvStandard(s_pred, self.mean_s, self.sd_s)
        X_pred = self.InvStandard(X_pred, self.mean_X, self.sd_X)
        return s_pred, X_pred

class VAEDesign(DesignColor):
    def __init__(self, mean_X, sd_X, mean_s, sd_s):
        super().__init__(mean_X, sd_X, mean_s, sd_s)
        self.name = "vae"

    def setupModel(self,):
        self.forward = Forward().to(device)
        self.model = cVAE_GSNN1(input_size=4, latent_dim=10).to(device)
        
    def loadModel(self, forward_state, model_state):
        self.forward.load_state_dict(forward_state)
        self.model.load_state_dict(model_state)
    
    def predict(self, color, nsamples=1):

        #网络输入
        input_color_tensor = torch.Tensor(color).to(device) #转化为Tensor

        # 预测结构

        s_pred, mu, logvar, temp = self.model.inference(input_color_tensor)
        X_pred = self.forward(s_pred)

        #转移出计算设备
        s_pred = s_pred.detach().cpu().numpy()
        X_pred = X_pred.detach().cpu().numpy()
        # 逆归一化
        s_pred = self.InvStandard(s_pred, self.mean_s, self.sd_s)
        X_pred = self.InvStandard(X_pred, self.mean_X, self.sd_X)
        return s_pred, X_pred

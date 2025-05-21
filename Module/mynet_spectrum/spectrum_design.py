# -*- coding: utf-8 -*-
import matlab
import matlab.engine
import sys
homepath = "../../"
sys.path.append(homepath)
import torch
import pandas as pd
import numpy as np
from Module.mynet_spectrum.pretrain import Forward
from Module.mynet_spectrum.main_v2 import Backward
import colour
from tools.getCData import getSpectral
#计算设备
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
####################################################################################
#目标颜色
    
class DesignSpectrum:
    def __init__(self, mean_X, sd_X, mean_s, sd_s):
        self.forward = Forward().to(device)
        self.backward = Backward().to(device)
        self.mean_X = mean_X
        self.sd_X = sd_X
        self.mean_s = mean_s
        self.sd_s = sd_s
    
    def loadModel(self, forward_state, backward_state):
        self.forward.load_state_dict(forward_state)
        self.backward.load_state_dict(backward_state)
    
    def loadIndex(self, index_SiH, index_SiO):
        self.index_SiH = index_SiH
        self.index_SiO = index_SiO
    
    def InvStandard(self, X, mean, sd):
        if mean is not None and sd is not None:
            X = X * sd + mean
        return X
    
    def predict(self, target):
        
        #网络输入
        input_target_tensor = torch.Tensor(target).to(device) #转化为Tensor
        #预测结构
        s_pred = self.backward(input_target_tensor, self.forward)
        #预测颜色
        X_pred = self.forward(s_pred)
        #转移出计算设备
        s_pred = s_pred.detach().cpu().numpy()
        X_pred = X_pred.detach().cpu().numpy()
        # 逆归一化
        s_pred = self.InvStandard(s_pred, self.mean_s, self.sd_s)
        return s_pred, X_pred
    
    def getSpectral(self, s):
        #建立matlab引擎
        engine = matlab.engine.start_matlab()
        engine.addpath(engine.genpath("E:/程序/python/reticolo_allege_v9"))
        engine.addpath(engine.genpath(homepath + 'tools'))
        
        R = getSpectral(s, self.index_SiH, self.index_SiO, engine)
        return R

    
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
# -*- coding: utf-8 -*-
import matlab
import matlab.engine
import sys 
homepath = "../../"
sys.path.append(homepath)
from Module.mynet_spectrum.main_v2 import Backward, InvStandard
from Module.mynet_spectrum.pretrain import Forward, Standard
from Module.mynet_spectrum.spectrum_design import DesignSpectrum
import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tools.MileServer import MileServer
from tools.ref2color import getColor

#设定计算设备
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
    
#读取数据，设计随机种子
seed = 42

data = np.array(pd.read_csv(homepath + "data/dataset/data_spectrum.csv", index_col=0))
X, s = data[:, :50], data[:, 50:]

mean_s, sd_s = np.mean(s, axis=0), np.std(s, axis=0)
s = Standard(s)
#划分训练测试
X_train, X_test, s_train, s_test = train_test_split(X, s, test_size=0.2,
                                                    random_state = seed)
X_test, s_test = torch.Tensor(X_test), torch.Tensor(s_test)

index_SiH = np.array(pd.read_csv(homepath + "data/index/index_SiH.csv"))
index_SiO = np.array(pd.read_csv(homepath + "data/index/index_SiO.csv"))

################################
forward_state = torch.load(homepath + "Module/mynet_spectrum/forward.pth", map_location=device)
backward_state = torch.load(homepath + "Module/mynet_spectrum/backward.pth", map_location=device)
dc = DesignSpectrum(None, None, mean_s, sd_s)
dc.loadIndex(index_SiH, index_SiO)
dc.loadModel(forward_state, backward_state)

s_pred, X_pred = dc.predict(X_test)
X_test = X_test.detach().cpu().numpy()

s_test = s_test.detach().cpu().numpy()
s_test = InvStandard(s_test, mean_s, sd_s)

##
#画图
def pic(idx):
    import matplotlib.pyplot as plt
    
    lamda = np.linspace(400, 750, 50)

    plt.figure(figsize=(6, 4), dpi=300)
    plt.plot(lamda, X_test[idx, :], "k", linewidth=3, label="True data")
    # plt.plot(lamda, X_pred[idx, :], "ro", markersize=5, label="Predicted data")
    
    plt.scatter(lamda, X_pred[idx, :], label="Predicted data", edgecolor=np.array([212, 131, 177]) / 255, 
                facecolor='none', s=50, linewidth=2, cmap='viridis', alpha=0.8)
    
    plt.xlabel("Wavelength (nm)", fontsize=25)
    plt.ylabel("Reflectance", fontsize=25)
    plt.xticks([400, 500, 600, 700], fontsize=20)
    plt.yticks([0.1, 0.3, 0.5, 0.7, 0.9], fontsize=20)
    plt.legend(frameon=False, fontsize=20)
    plt.tick_params(direction="in", top=True, right=True, size=6, width=3)
    ax = plt.gca()
    ax.spines["bottom"].set_linewidth(3)
    ax.spines["top"].set_linewidth(3)
    ax.spines["left"].set_linewidth(3)
    ax.spines["right"].set_linewidth(3)
    plt.show()

index = np.random.randint(0, 7000)
pic(5878)
result = []
    





        

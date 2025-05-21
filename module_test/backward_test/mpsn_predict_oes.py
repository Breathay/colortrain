# -*- coding: utf-8 -*-
import matlab
import matlab.engine
import sys 
homepath = "../../"
sys.path.append(homepath)
from Module.forward_xyz.pretrain_oes import Standard, InvStandard
from Module.ColorDesign import MPSNDesign
from Module.forward_xyz.pretrain_oes import Forward as Forward_OES
from Module.mynet_xyz.main_oes import Backward as Backward_OES    
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

data = np.array(pd.read_csv(homepath + "data/dataset_oes/data_w.csv", index_col=0))
X, s = data[:, :3], data[:, 3:]

mean_X, sd_X = np.mean(X, axis=0), np.std(X, axis=0)
X = Standard(X)

mean_s, sd_s = np.mean(s, axis=0), np.std(s, axis=0)
s = Standard(s)
#划分训练测试
X_train, X_test, s_train, s_test = train_test_split(X, s, test_size=0.2,
                                                    random_state = seed)
#划分测试数据
start, end = 0, 100
X_test, s_test = X_test[start: end], s_test[start: end]
X_test, s_test = torch.Tensor(X_test), torch.Tensor(s_test)

index_SiH = np.array(pd.read_csv(homepath + "data/index/index_SiH.csv"))
index_SiO = np.array(pd.read_csv(homepath + "data/index/index_SiO.csv"))

################################
forward_state = torch.load(homepath + "Module/forward_xyz/forward_oes.pth", map_location=device)
backward_state = torch.load(homepath + "Module/mynet_xyz/model_oes.pth", map_location=device)
dc = MPSNDesign(mean_X, sd_X, mean_s, sd_s)
dc.forward = Forward_OES().to(device)
dc.backward = Backward_OES().to(device)
dc.loadIndex(index_SiH, index_SiO)
dc.loadModel(forward_state, backward_state)

s_pred, X_pred = dc.predict(X_test)
X_test = X_test.detach().cpu().numpy()
X_test = InvStandard(X_test, mean_X, sd_X)

s_test = s_test.detach().cpu().numpy()
s_test = InvStandard(s_test, mean_s, sd_s)

##
#画图
def pic():
    import matplotlib.pyplot as plt
    
    temp1 = X_test[:, 0] - np.mean(X_test, axis=0)[0]
    temp2 = X_pred[:, 0] - np.mean(X_pred, axis=0)[0]
    fenzi = np.sum(temp1 * temp2)
    fenmu = np.sqrt(np.sum(np.square(temp1))) * np.sqrt(np.sum(np.square(temp2)))
    r = np.square(fenzi / fenmu)
    
    plt.figure(figsize=(6, 4), dpi=300)
    plt.scatter(X_test[:, 0], X_pred[:, 0], s=200 * np.ones_like(X_test[:, 0]), 
                c=np.random.rand(len(X_test[:, 0])), cmap='viridis', alpha=0.5)
    plt.plot(X_test[:, 0], X_test[:, 0], "k", linewidth=3)
    plt.xlabel("True x", fontsize=25)
    plt.ylabel("Predict x", fontsize=25)
    plt.tick_params(direction="in", top=True, right=True, size=6, width=3)
    ax = plt.gca()
    ax.spines["bottom"].set_linewidth(3)
    ax.spines["top"].set_linewidth(3)
    ax.spines["left"].set_linewidth(3)
    ax.spines["right"].set_linewidth(3)
    plt.xticks([0.1, 0.3, 0.5, 0.7], fontsize=20)
    plt.yticks([0.1, 0.3, 0.5, 0.7], fontsize=20)
    plt.text(0.21, 0.5, "$R^2={:.3f}$".format(r), fontsize=20)
    plt.show()
    
    temp1 = X_test[:, 1] - np.mean(X_test, axis=0)[1]
    temp2 = X_pred[:, 1] - np.mean(X_pred, axis=0)[1]
    fenzi = np.sum(temp1 * temp2)
    fenmu = np.sqrt(np.sum(np.square(temp1))) * np.sqrt(np.sum(np.square(temp2)))
    r = np.square(fenzi / fenmu)
    
    plt.figure(figsize=(6, 4), dpi=300)
    plt.scatter(X_test[:, 1], X_pred[:, 1], s=200 * np.ones_like(X_test[:, 0]), 
                c=np.random.rand(len(X_test[:, 0])), cmap='viridis', alpha=0.5)
    plt.plot(X_test[:, 1], X_test[:, 1], "k", linewidth=3)
    plt.xlabel("True y", fontsize=25)
    plt.ylabel("Predict y", fontsize=25)
    plt.tick_params(direction="in", top=True, right=True, size=6, width=3)
    ax = plt.gca()
    ax.spines["bottom"].set_linewidth(3)
    ax.spines["top"].set_linewidth(3)
    ax.spines["left"].set_linewidth(3)
    ax.spines["right"].set_linewidth(3)
    plt.xticks([0.1, 0.3, 0.5, 0.7], fontsize=20)
    plt.yticks([0.1, 0.3, 0.5, 0.7], fontsize=20)
    plt.text(0.21, 0.5, "$R^2={:.3f}$".format(r), fontsize=20)
    plt.show()
    
    temp1 = X_test[:, 2] - np.mean(X_test, axis=0)[2]
    temp2 = X_pred[:, 2] - np.mean(X_pred, axis=0)[2]
    fenzi = np.sum(temp1 * temp2)
    fenmu = np.sqrt(np.sum(np.square(temp1))) * np.sqrt(np.sum(np.square(temp2)))
    r = np.square(fenzi / fenmu)
    
    plt.figure(figsize=(6, 4), dpi=300)
    plt.scatter(X_test[:, 2], X_pred[:, 2], s=200 * np.ones_like(X_test[:, 0]), 
                c=np.random.rand(len(X_test[:, 0])), cmap='viridis', alpha=0.5)
    plt.plot(X_test[:, 2], X_test[:, 2], "k", linewidth=3)
    plt.xlabel("True Y", fontsize=25)
    plt.ylabel("Predict Y", fontsize=25)
    plt.tick_params(direction="in", top=True, right=True, size=6, width=3)
    ax = plt.gca()
    ax.spines["bottom"].set_linewidth(3)
    ax.spines["top"].set_linewidth(3)
    ax.spines["left"].set_linewidth(3)
    ax.spines["right"].set_linewidth(3)
    plt.xticks([0.1, 0.3, 0.5, 0.7], fontsize=20)
    plt.yticks([0.1, 0.3, 0.5, 0.7], fontsize=20)
    plt.text(0.21, 0.5, "$R^2={:.3f}$".format(r), fontsize=20)
    plt.show()

pic()
    
    





        

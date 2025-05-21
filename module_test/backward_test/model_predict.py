# -*- coding: utf-8 -*-
import matlab
import matlab.engine
import sys 
homepath = "../../"
sys.path.append(homepath)
from tools.DataLoader import Standard, InvStandard
from Module.ColorDesign import MPSNDesign, CGANDesign, VAEDesign, MDNDesign, TNNDesign
import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tools.MileServer import MileServer
from tools.ref2color import getColor
import os

#设定计算设备
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
seed = 42

def read_data(start=0, end=100):
    #读取数据，设计随机种子

    data = np.array(pd.read_csv(homepath + "data/dataset/data_w_new.csv", index_col=0))
    X, s = data[:, 3:6], data[:, 6:]

    X = X / np.sum(X, axis=1, keepdims=True)
    X = X[:, :2]
    mean_X, sd_X = np.mean(X, axis=0), np.std(X, axis=0)
    X = Standard(X)

    mean_s, sd_s = np.mean(s, axis=0), np.std(s, axis=0)
    s = Standard(s)
    #划分训练测试
    X_train, X_test, s_train, s_test = train_test_split(X, s, test_size=0.2,
                                                        random_state = seed)
    #划分测试数据
    X_test, s_test = X_test[start: end], s_test[start: end]
    X_test, s_test = torch.Tensor(X_test), torch.Tensor(s_test)

    index_SiH = np.array(pd.read_csv(homepath + "data/index/index_SiH.csv"))
    index_SiO = np.array(pd.read_csv(homepath + "data/index/index_SiO.csv"))

    return X_test, s_test, mean_X, sd_X, mean_s, sd_s, index_SiH, index_SiO

def load_forward_model():
    forward_state = torch.load(homepath + "Module/forward_xyz/forward_xy.pth", map_location=device)
    return forward_state

################################
def model_predict(dc, X_test, s_test, index_SiH, index_SiO, mean_X, sd_X, mean_s, sd_s, forward_state):
    
    backward_state = torch.load(homepath + "Module/{}_xyz/model.pth".format(dc.name), map_location=device)
    dc.setupModel()
    dc.loadIndex(index_SiH, index_SiO)
    dc.loadModel(forward_state, backward_state)

    s_pred, X_pred = dc.predict(X_test, 100)
    X_test = X_test.detach().cpu().numpy()

    s_test = s_test.detach().cpu().numpy()
    s_test = InvStandard(s_test, mean_s, sd_s)

    X_test = InvStandard(X_test, mean_X, sd_X)

    return s_pred, X_pred, s_test, X_test

def color_predict(dc, input_color, index_SiH, index_SiO, forward_state):
    
    backward_state = torch.load(homepath + "Module/{}_xyz/model.pth".format(dc.name), map_location=device)
    dc.setupModel()
    dc.loadIndex(index_SiH, index_SiO)
    dc.loadModel(forward_state, backward_state)

    s_pred, X_pred = dc.predict(input_color)

    return s_pred, X_pred

##
#画图
def pic(X_test, X_pred):
    import matplotlib.pyplot as plt
    
    temp1 = X_test[:, 0] - np.mean(X_test, axis=0)[0]
    temp2 = X_pred[:, 0] - np.mean(X_pred, axis=0)[0]
    fenzi = np.sum(temp1 * temp2)
    fenmu = np.sqrt(np.sum(np.square(temp1))) * np.sqrt(np.sum(np.square(temp2)))
    r = np.square(fenzi / fenmu)

    plt.figure(figsize=(10, 7), dpi=300)
    plt.scatter(X_test[:, 0], X_pred[:, 0], s=200 * np.ones_like(X_test[:, 0]), 
                c=np.random.rand(len(X_test[:, 0])), cmap='viridis', alpha=0.5)
    plt.plot(X_test[:, 0], X_test[:, 0], "k", linewidth=3)
    plt.xlabel("True $x$", fontsize=50)
    plt.ylabel("Predict $x$", fontsize=50)
    plt.xticks(fontsize=40)
    plt.tick_params(direction="in", width=2, size=6, pad=15)
    plt.yticks(fontsize=40)
    plt.text(0.25, 0.45, "$R^2={:.3f}$".format(r), fontsize=50)
    ax = plt.gca()
    ax.spines["bottom"].set_linewidth(4)
    ax.spines["top"].set_linewidth(4)
    ax.spines["left"].set_linewidth(4)
    ax.spines["right"].set_linewidth(4)
    plt.show()
        
    temp1 = X_test[:, 1] - np.mean(X_test, axis=0)[1]
    temp2 = X_pred[:, 1] - np.mean(X_pred, axis=0)[1]
    fenzi = np.sum(temp1 * temp2)
    fenmu = np.sqrt(np.sum(np.square(temp1))) * np.sqrt(np.sum(np.square(temp2)))
    r = np.square(fenzi / fenmu)

    plt.figure(figsize=(10, 7), dpi=300)
    plt.scatter(X_test[:, 1], X_pred[:, 1], s=200 * np.ones_like(X_test[:, 0]), 
                c=np.random.rand(len(X_test[:, 0])), cmap='viridis', alpha=0.5)
    plt.plot(X_test[:, 1], X_test[:, 1], "k", linewidth=3)
    plt.xlabel("True $x$", fontsize=50)
    plt.ylabel("Predict $x$", fontsize=50)
    plt.xticks(fontsize=40)
    plt.tick_params(direction="in", width=2, size=6)
    plt.yticks(fontsize=40)
    plt.text(0.2, 0.45, "$R^2={:.3f}$".format(r), fontsize=50)
    ax = plt.gca()
    ax.spines["bottom"].set_linewidth(4)
    ax.spines["top"].set_linewidth(4)
    ax.spines["left"].set_linewidth(4)
    ax.spines["right"].set_linewidth(4)
    plt.show()

if __name__ == "__main__":
    import time
    start = time.time() 

    X_test, s_test, mean_X, sd_X, mean_s, sd_s, index_SiH, index_SiO = read_data()
    forward_state = load_forward_model()

    mpsn = MPSNDesign(mean_X, sd_X, mean_s, sd_s)
    cgan = CGANDesign(mean_X, sd_X, mean_s, sd_s)
    vae = VAEDesign(mean_X, sd_X, mean_s, sd_s)
    mdn = MDNDesign(mean_X, sd_X, mean_s, sd_s)
    tnn = TNNDesign(mean_X, sd_X, mean_s, sd_s)

    mpsn_s_pred, mpsn_X_pred, mpsn_s_test, mpsn_X_test = model_predict(mpsn, X_test, s_test, index_SiH, index_SiO, mean_X, sd_X, mean_s, sd_s, forward_state)
    cgan_s_pred, cgan_X_pred, cgan_s_test, cgan_X_test = model_predict(cgan, X_test, s_test, index_SiH, index_SiO, mean_X, sd_X, mean_s, sd_s, forward_state)
    vae_s_pred, vae_X_pred, vae_s_test, vae_X_test = model_predict(vae, X_test, s_test, index_SiH, index_SiO, mean_X, sd_X, mean_s, sd_s, forward_state)
    mdn_s_pred, mdn_X_pred, mdn_s_test, mdn_X_test = model_predict(mdn, X_test, s_test, index_SiH, index_SiO, mean_X, sd_X, mean_s, sd_s, forward_state)
    tnn_s_pred, tnn_X_pred, tnn_s_test, tnn_X_test = model_predict(tnn, X_test, s_test, index_SiH, index_SiO, mean_X, sd_X, mean_s, sd_s, forward_state)
    
    pic(mpsn_X_test, mpsn_X_pred)
    pic(cgan_X_test, cgan_X_pred)
    pic(vae_X_test, vae_X_pred)
    pic(tnn_X_test, tnn_X_pred)
    
    print(mpsn_s_pred)
    print(tnn_s_pred)
    
    end = time.time()

    
    





        

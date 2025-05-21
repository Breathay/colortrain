# -*- coding: utf-8 -*-
import matlab
import matlab.engine
import sys 
homepath = "../../"
from main_v1 import Backward
from pretrain import Forward, Standard, InvStandard
import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tools.MileServer import MileServer
#建立matlab引擎
engine = matlab.engine.start_matlab()
engine.addpath(engine.genpath('../reticolo_allege_v9'))
engine.addpath(engine.genpath('../'))
#读取数据，设计随机种子
seed = 42

data = np.array(pd.read_csv(homepath + "data/dataset/data_old.csv", index_col=0))
X, s = data[:, :3], data[:, 3:]

standard = np.loadtxt(homepath + "data/standard/standard.txt")
mean_X, sd_X = standard[:, 2], standard[:, 3]

X = Standard(X)

index_SiH = np.array(pd.read_csv(homepath + "data/index/index_SiH.csv"))
index_SiO = np.array(pd.read_csv(homepath + "data/index/index_SiO.csv"))
#归一化参数
mean_s, sd_s = np.mean(s, axis=0), np.std(s, axis=0)
s = Standard(s)
#划分训练测试
X_train, X_test, s_train, s_test = train_test_split(X, s, test_size=0.2,
                                                    random_state = seed)
#划分测试数据
start, end = 0, 500
X_test, s_test = X_test[start: end], s_test[start: end]
X_test, s_test = torch.Tensor(X_test), torch.Tensor(s_test)
#设定计算设备
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
#读取正向模型
forward = Forward().to(device)
forward_state = torch.load("forward.pth", map_location=device)
forward.load_state_dict(forward_state)
#读取反向模型
backward = Backward().to(device)
backward_state = torch.load("backward.pth", map_location=device)
backward.load_state_dict(backward_state)
#转移到计算设备
X_test, s_test = X_test.to(device), s_test.to(device)
#预测
s_pred = backward(X_test, forward)
X_pred = forward(s_pred)
#转移出计算设备
X_pred = X_pred.cpu().detach().numpy()
s_pred = s_pred.cpu().detach().numpy()
#反归一化
s_pred = InvStandard(s_pred, mean_s, sd_s)
X_test, s_test = X_test.cpu().numpy(), s_test.cpu().numpy()
s_test = InvStandard(s_test, mean_s, sd_s)
#画图
def pic():
    import matplotlib.pyplot as plt
    
    temp1 = X_test[:, 0] - np.mean(X_test, axis=0)[0]
    temp2 = X_pred[:, 0] - np.mean(X_pred, axis=0)[0]
    fenzi = np.sum(temp1 * temp2)
    fenmu = np.sqrt(np.sum(np.square(temp1))) * np.sqrt(np.sum(np.square(temp2)))
    r = np.square(fenzi / fenmu)
    
    plt.figure(figsize=(6, 4), dpi=300)
    plt.plot(X_test[:, 0], X_pred[:, 0], "k*", markersize=12)
    plt.plot(X_test[:, 0], X_test[:, 0], "r", linewidth=2.5)
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
    plt.plot(X_test[:, 1], X_pred[:, 1], "k*", markersize=12)
    plt.plot(X_test[:, 1], X_test[:, 1], "r", linewidth=2.5)
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
    plt.plot(X_test[:, 2], X_pred[:, 2], "k*", markersize=12)
    plt.plot(X_test[:, 2], X_test[:, 2], "r", linewidth=2.5)
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

# import colour
# from colour import SpectralDistribution, SDS_ILLUMINANTS
# from scipy.interpolate import CubicSpline

# def getColor(R):
#     l = np.linspace(400, 750, len(R))
#     cs = CubicSpline(l, R)
#     lamda = np.arange(400, 750, 5)
#     data = cs(lamda)
    
#     d = dict(np.c_[lamda, data])
    
#     sd = SpectralDistribution(d)
#     illuminant = SDS_ILLUMINANTS['D65']
#     rgb = colour.convert(sd, 'Spectral Distribution', 'sRGB', verbose={'mode': 'Short'},
#                          sd_to_XYZ={'illuminant': illuminant})
#     for i in range(len(rgb)):
#         if rgb[i] < 0:
#             rgb[i] = 0
#         if rgb[i] > 1:
#             rgb[i] = 1
#     return rgb

# print(np.mean(np.square(s_test - s_pred)))
# pic()
# result = []
# for i in range(start, end):
#     p, w, ww, d = s_pred[i]
#     # p = (w1 + w2 + w3 + w4) * 2
#     # w = p - 2 * w1;
#     # ww = 2 * w4;
#     # d = w2;
#     R = engine.eff_eval(matlab.double([p]), 
#                         matlab.double([w]), 
#                         matlab.double([ww]), 
#                         matlab.double([d]), 
#                         matlab.double(index_SiH.tolist()),
#                         matlab.double(index_SiO.tolist()),
#                         matlab.double([1]))
#     R = np.array(R)[0]
#     color = getColor(R)
#     result.append(color)
#     print(i)

# result = np.array(result)

# np.savetxt("data/target_" + str(seed) + ".txt", X_test, delimiter="\t")
# np.savetxt("data/predict_"+ str(seed) + ".txt", X_pred, delimiter="\t")
# np.savetxt("data/result_" + str(seed) + ".txt", result, delimiter="\t")

# server = MileServer()
# server.setEmileContent('Predict Finished', str(end - start) + 'second used')
# server.sendEmile()   

    
    





        

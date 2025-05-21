# -*- coding: utf-8 -*-
import sys 
homepath = "../"
sys.path.append(homepath)
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Module.forward_xyz.pretrain_oes import Forward, Standard
from sklearn.model_selection import train_test_split

if __name__ == "__main__":

    plt.rcParams['font.sans-serif']=['Arial']

    data = np.array(pd.read_csv(homepath + "data/dataset_oes/data_w.csv").iloc[:, 1:])
    X, s = data[:, :3], data[:, 3:]

    # standard = np.loadtxt(homepath + "data/standard/standard.txt")
    # mean_X, sd_X = standard[:, 2], standard[:, 3]

    X = Standard(X,)

    s = Standard(s)
    X_train, X_test, s_train, s_test = train_test_split(X, s, test_size=0.2,
                                                            random_state = 42)

    start, end = 0, -1
    X_test, s_test = torch.Tensor(X_test[start: end]), torch.Tensor(s_test[start: end])

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    #读取模型参数
    forward = Forward().to(device)
    forward_state = torch.load(homepath + "Module/forward_xyz/forward_oes.pth", map_location=device)
    forward.load_state_dict(forward_state)

    #进入计算器件
    X_test, s_test = X_test.to(device), s_test.to(device)

    #正向计算
    X_pred = forward(s_test)
    X_pred = X_pred.cpu().detach().numpy()

    #从器件中转移出
    X_test, s_test = X_test.cpu().numpy(), s_test.cpu().numpy()
    #划分测试数据

    target = X_test[start: end]
    result = X_pred[start: end]
    print(result)
    print(target)
    ##################################
    temp1 = target[:, 0] - np.mean(target, axis=0)[0]
    temp2 = result[:, 0] - np.mean(result, axis=0)[0]
    fenzi = np.sum(temp1 * temp2)
    fenmu = np.sqrt(np.sum(np.square(temp1))) * np.sqrt(np.sum(np.square(temp2)))
    r = np.square(fenzi / fenmu)
    plt.figure(figsize=(6, 4), dpi=300)
    plt.plot(target[:, 0], result[:, 0], "k*", markersize=12)
    plt.plot(target[:, 0], target[:, 0], "r", linewidth=2.5)
    plt.xlabel("True x", fontsize=25)
    plt.ylabel("Predict x", fontsize=25)
    plt.xticks([0.1, 0.3, 0.5, 0.7], fontsize=20)
    plt.yticks([0.1, 0.3, 0.5, 0.7], fontsize=20)
    plt.text(0.2, 0.5, "$R^2={:.3f}$".format(r), fontsize=20)
    plt.tick_params(direction="in", top=True, right=True, size=6, width=3)
    ax = plt.gca()
    ax.spines["bottom"].set_linewidth(3)
    ax.spines["top"].set_linewidth(3)
    ax.spines["left"].set_linewidth(3)
    ax.spines["right"].set_linewidth(3)
    plt.show()
    ############################
    temp1 = target[:, 1] - np.mean(target, axis=0)[1]
    temp2 = result[:, 1] - np.mean(result, axis=0)[1]
    fenzi = np.sum(temp1 * temp2)
    fenmu = np.sqrt(np.sum(np.square(temp1))) * np.sqrt(np.sum(np.square(temp2)))
    r = np.square(fenzi / fenmu)

    plt.figure(figsize=(6, 4), dpi=300)
    plt.plot(target[:, 1], result[:, 1], "k*", markersize=12)
    plt.plot(target[:, 1], target[:, 1], "r", linewidth=2.5)
    plt.xlabel("True y", fontsize=25)
    plt.ylabel("Predict y", fontsize=25)
    plt.xticks([0.1, 0.3, 0.5, 0.7], fontsize=20)
    plt.yticks([0.1, 0.3, 0.5, 0.7], fontsize=20)
    plt.text(0.2, 0.5, "$R^2={:.3f}$".format(r), fontsize=20)
    plt.tick_params(direction="in", top=True, right=True, size=6, width=3)
    ax = plt.gca()
    ax.spines["bottom"].set_linewidth(3)
    ax.spines["top"].set_linewidth(3)
    ax.spines["left"].set_linewidth(3)
    ax.spines["right"].set_linewidth(3)
    plt.show()
    ###########################
    temp1 = target[:, 2] - np.mean(target, axis=0)[2]
    temp2 = result[:, 2] - np.mean(result, axis=0)[2]
    fenzi = np.sum(temp1 * temp2)
    fenmu = np.sqrt(np.sum(np.square(temp1))) * np.sqrt(np.sum(np.square(temp2)))
    r = np.square(fenzi / fenmu)

    plt.figure(figsize=(6, 4), dpi=300)
    plt.plot(target[:, 2], result[:, 2], "k*", markersize=12)
    plt.plot(target[:, 2], target[:, 2], "r", linewidth=2.5)
    plt.xlabel("True y", fontsize=25)
    plt.ylabel("Predict y", fontsize=25)
    plt.xticks([0.1, 0.3, 0.5], fontsize=20)
    plt.yticks([0.1, 0.3, 0.5], fontsize=20)
    plt.text(0.1, 0.37, "$R^2={:.3f}$".format(r), fontsize=20)
    plt.tick_params(direction="in", top=True, right=True, size=6, width=3)
    ax = plt.gca()
    ax.spines["bottom"].set_linewidth(3)
    ax.spines["top"].set_linewidth(3)
    ax.spines["left"].set_linewidth(3)
    ax.spines["right"].set_linewidth(3)
    plt.show()







        
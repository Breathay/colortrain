# -*- coding: utf-8 -*-
import sys 
homepath = "../../"
from main import Backward
from pretrain import Forward, Standard, InvStandard
import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tools.MileServer import MileServer

data = np.array(pd.read_csv(homepath + "data/dataset/data_w.csv", index_col=0))
X, s = data[:, 3:6], data[:, 6:]

X = Standard(X)
mean_s, sd_s = np.mean(s, axis=0), np.std(s, axis=0)
s = Standard(s)

#划分训练测试
X_train, X_test, s_train, s_test = train_test_split(X, s, test_size=0.2,
                                                    random_state = 42)
#划分测试数据
start, end = 0, 100
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

X_test = X_test.to(device)

y_pred = backward(X_test)





























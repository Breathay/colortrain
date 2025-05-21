# -*- coding: utf-8 -*-
import sys 
homepath = "../"
sys.path.append(homepath)
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Module.mynet_spectrum.pretrain import Forward, Standard
from sklearn.model_selection import train_test_split

plt.rcParams['font.sans-serif']=['Arial']

data = np.array(pd.read_csv(homepath + "data/dataset/data_spectrum.csv").iloc[:, 1:])
X, s = data[:, :50], data[:, 50:]

# standard = np.loadtxt(homepath + "data/standard/standard.txt")
# mean_X, sd_X = standard[:, 2], standard[:, 3]

# X = Standard(X,)

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
forward_state = torch.load(homepath + "Module/mynet_spectrum/forward.pth", map_location=device)
forward.load_state_dict(forward_state)

#进入计算器件
X_test, s_test = X_test.to(device), s_test.to(device)

#正向计算
X_pred = forward(s_test)
X_pred = X_pred.cpu().detach().numpy()

#从器件中转移出
X_test, s_test = X_test.cpu().numpy(), s_test.cpu().numpy()

target = X_test[start: end]
result = X_pred[start: end]
# print(result)
# print(target)
##################################

lamda = np.linspace(400, 750, 50)

idx = np.random.randint(0, 7000)
idx = 5878
print(idx)
plt.figure(figsize=(6, 4), dpi=300)
plt.plot(lamda, target[idx, :], "k", linewidth=3, label="True data")
plt.scatter(lamda, result[idx, :], label="Predicted data", edgecolor=np.array([212, 131, 177]) / 255, 
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
############################







        
# -*- coding: utf-8 -*-
import numpy as np
import scipy.io as scio
import pandas as pd

# 读取mat文件，并转换为numpy数组
def read_mat_file(file_path):
    data = scio.loadmat(file_path)
    return data

data = read_mat_file("RCWA_xyY_all.mat")['data_rcwa_xyY']

# 将xyY转换为XYZ
xyY = data[:, 4:]
s = data[:, :4]
# 将xyY转换为XYZ
def xyY_to_XYZ(xyY):
    X = xyY[:, 0] * xyY[:, 2] / xyY[:, 1]
    Y = xyY[:, 2]
    Z = (1 - xyY[:, 0] - xyY[:, 1]) * xyY[:, 2] / xyY[:, 1]
    return np.stack([X, Y, Z], axis=-1)

XYZ = xyY_to_XYZ(xyY)

result = np.concatenate([XYZ, s], axis=-1)

# 保存为csv文件
pd.DataFrame(result).to_csv("data_w.csv")


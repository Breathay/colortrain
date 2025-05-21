# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt

index_SiH = np.array(pd.read_excel("SiH-不同厚度.xlsx"))[:, :]
l = index_SiH[:, 3]
n = index_SiH[:, 4]
k = index_SiH[:, 5]

lamda = np.linspace(400, 750, 50)

tn = interpolate.splrep(l, n)
tk = interpolate.splrep(l, k)

n0 = interpolate.splev(lamda, tn)
k0 = interpolate.splev(lamda, tk)

data = np.c_[lamda, n0]
data = np.c_[data, k0]

df = pd.DataFrame(data, columns=["lamda", "n", "k"])
df.to_csv("index_SiH_huxi_110.csv", index=0)


plt.figure(figsize=(10, 7))
plt.plot(lamda, n0, "k", linewidth=3, label="n")
plt.plot(lamda, k0, "r", linewidth=3, label="k")
plt.legend()
plt.show()


# df = np.array(pd.read_csv("index_SiO.csv"))
# n = df[:, 1]
# k = df[:, 2]

# plt.figure(figsize=(10, 7))
# plt.plot(lamda, n, "b*-", linewidth=3, label="old_n")
# plt.plot(lamda, k, "g*-", linewidth=3, label="old_k")
# plt.legend()
# plt.show()


# df = np.loadtxt("SiO_taiwan.txt", skiprows=1)
# lamda = np.linspace(400, 750, 50)

# data = np.array(df)
# l = data[:, 0]
# n = data[:, 1]
# k = data[:, 2]

# tn = interpolate.splrep(l, n)
# tk = interpolate.splrep(l, k)

# n0 = interpolate.splev(lamda, tn)
# k0 = interpolate.splev(lamda, tk)

# data = np.c_[lamda, n0]
# data = np.c_[data, k0]

# df = pd.DataFrame(data, columns=["lamda", "n", "k"])
# df.to_csv("index_SiO_taiwan.csv", index=0)
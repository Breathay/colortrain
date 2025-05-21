# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import cv2

homepath = "../"

data = np.array(pd.read_csv(homepath + "data/dataset/data_w.csv").iloc[:, 1:])

X, s = data[:, :3], data[:, 6:]


X_max = np.max(X, axis=0)
X_min = np.min(X, axis=0)
X_mean = np.mean(X, axis=0)
X_sd = np.std(X, axis=0)

print(X_max, X_min, X_mean, X_sd)


res = np.c_[X_max, X_min, X_mean, X_sd]
np.savetxt(homepath + "data/standard/standard.txt", res)

plt.rcParams['font.sans-serif']=['Arial']

r, g, b = X[:, 0] * 255, X[:, 1] * 255, X[:, 2] * 255

r, g, b = r.astype(np.uint8), g.astype(np.uint8), b.astype(np.uint8)

fig = plt.figure(dpi=300)

ax = fig.add_subplot(111, projection='3d')
for x, c, z in zip([r, g, b], ['r', 'g', 'b'], [30, 20, 10]):
    xs = np.arange(256)
    ys = cv2.calcHist([x], [0], None, [256], [0,256])
    cs = [c] * len(xs)
    cs[0] = 'c'
    ax.bar(xs, ys.ravel(), zs=z, zdir='y', color=cs, alpha=0.8)

ax.set_zlabel('Conut', fontsize=20)
ax.set_yticks([10, 20, 30], ["B", "G", "R"], fontsize=13)
ax.set_zticks([0, 100, 200, 300], fontsize=20)
ax.set_xticks([0, 50, 100, 150, 200], fontsize=20)
ax.set_xlim(0, 200)
ax.set_zlim(0, 300)
plt.tick_params(direction="in", width=2, size=6)

plt.show()
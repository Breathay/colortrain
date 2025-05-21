# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import cv2

S_mpsn = np.loadtxt("S_mpsn")
S_mpsn_1 = np.loadtxt("S_mpsn_1")
S_mpsn_5 = np.loadtxt("S_mpsn_5")
S_mpsn_10 = np.loadtxt("S_mpsn_10")
S_mpsn_20 = np.loadtxt("S_mpsn_20")

plt.rcParams['font.sans-serif']=['Arial']
plt.figure(dpi=300)
label_size = 25
tick_size = 20

i = 1

color = [np.array([212, 131, 177]) / 255, np.array([98, 183, 150]) / 255, 
         np.array([107, 155, 183]) / 255, np.array([234, 188, 142]) / 255]

result = [S_mpsn_1[:, i], S_mpsn[:, i], S_mpsn_10[:, 3], S_mpsn_20[:, i]]

fig = plt.figure(dpi=300)

ax = fig.add_subplot(111, projection='3d')
for x, c, z in zip(result, color, [10, 20, 30, 40]):
    xs = np.arange(256)
    
    x[np.where(x < 0)] = np.random.uniform(S_mpsn[:, i].min(), S_mpsn[:, i].max())
    
    ys = cv2.calcHist([x.astype(np.uint8)], [0], None, [256], [0, 256])
    cs = [c] * len(xs)
    cs[0] = 'c'
    ax.bar(xs, ys.ravel(), zs=z, zdir='y', color=cs, alpha=0.8, width=5)

ax.set_zlabel('Conut', fontsize=18)
ax.set_xlabel('Width (nm)', fontsize=18)
ax.set_ylabel('Sampling number', fontsize=18)
ax.set_yticks([10, 20, 30, 40], ["1", "5", "10", "20"], fontsize=13)
ax.set_zticks([0, 5, 10], fontsize=20)
ax.set_xticks([0, 50, 100, 150, 200, 250], fontsize=20)
ax.set_xlim(0, 250)
ax.set_zlim(0, 10)
plt.tick_params(direction="in", width=2, size=6)

plt.show()
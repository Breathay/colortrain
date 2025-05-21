# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import cv2

S_mpsn = np.loadtxt("S_mpsn.txt")
S_cgan = np.loadtxt("S_cgan.txt")
S_vae = np.loadtxt("S_vae.txt")
S_mdn = np.loadtxt("S_mdn.txt")


plt.rcParams['font.sans-serif']=['Arial']

# plt.figure(dpi=300)
# label_size = 25
# tick_size = 20
# leg = ["w1", "w2", "w3", "w4"]

# i = 2

# plt.hist(S_mpsn[:, i], label="MPSN", bins=40)

# plt.xticks(fontsize=tick_size)
# plt.yticks(fontsize=tick_size)
# # plt.ylim(0, 500)
# plt.xlim(-100, 600)

# plt.xlabel("Predicted (nm)", fontsize=label_size)
# plt.ylabel("Count", fontsize=label_size)
# plt.tick_params(direction="in", top=True, right=True, size=6, width=2)
# plt.legend(fontsize=15, loc="best", frameon=False)
# ax = plt.gca()
# ax.spines["bottom"].set_linewidth(2)
# ax.spines["top"].set_linewidth(2)
# ax.spines["left"].set_linewidth(2)
# ax.spines["right"].set_linewidth(2)
# plt.show()

i = 1

color = [np.array([212, 131, 177]) / 255, np.array([98, 183, 150]) / 255, 
         np.array([107, 155, 183]) / 255, np.array([234, 188, 142]) / 255]

S_mpsn = np.where(S_mpsn > 200, 200, S_mpsn)

result = [S_mdn[:, i], S_cgan[:, i], S_vae[:, i], S_mpsn[:, i]]

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
ax.set_yticks([10, 20, 30, 40], ["MDN", "CGAN", "VAE", "MPSN"], fontsize=13)
ax.set_zticks([0, 10, 20, 30, 40], fontsize=20)
ax.set_xticks([0, 50, 100, 150, 200,], fontsize=20)
ax.set_xlim(0, 200)
ax.set_zlim(0, 40)
plt.tick_params(direction="in", width=2, size=6)

plt.show()
# -*- coding: utf-8 -*-
import sys 
homepath = "../"
sys.path.append(homepath)
import numpy as np
import matplotlib.pyplot as plt

module_name = "mynet_spectrum"

plt.rcParams['font.sans-serif']=['Arial']

loss1 = np.loadtxt(homepath + "Module/" + module_name + "/loss_forward.txt")
plt.figure(figsize=(6, 4), dpi=300)
plt.plot(loss1, linewidth=3)
plt.xlabel("Epoch", fontsize=25)
plt.xticks([0, 40, 80, 120], fontsize=20)
plt.yticks(fontsize=20)
plt.xlim(-5, 150)
plt.yscale("log")
plt.ylabel("Mean Square Error", fontsize=25)
plt.tick_params(direction="in", top=True, right=True, size=6, width=2.5)
ax = plt.gca()
ax.spines["bottom"].set_linewidth(3)
ax.spines["top"].set_linewidth(3)
ax.spines["left"].set_linewidth(3)
ax.spines["right"].set_linewidth(3)
plt.show()

loss1 = np.loadtxt(homepath + "Module/" + module_name + "/loss_backward.txt")
plt.figure(figsize=(6, 4), dpi=300)
plt.plot(loss1, linewidth=3)


plt.yscale("log")

plt.xticks(fontsize=20)
plt.yticks([0.0006, 0.001, 0.002, 0.003], fontsize=20)

plt.xlabel("Epoch", fontsize=25)
plt.ylabel("Mean Square Error", fontsize=25)

plt.tick_params(direction="in", top=True, right=True, size=6, width=2.5)

ax = plt.gca()
ax.spines["bottom"].set_linewidth(3)
ax.spines["top"].set_linewidth(3)
ax.spines["left"].set_linewidth(3)
ax.spines["right"].set_linewidth(3)
plt.show()

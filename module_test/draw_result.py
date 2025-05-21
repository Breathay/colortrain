# -*- coding: utf-8 -*-
import sys
homepath = "../"
sys.path.append(homepath)
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif']=['Arial']
#Predict X Y
seed = 42
target = np.loadtxt("module_mynet_xyz_test2/target2_" + str(seed) + ".txt")
predict = np.loadtxt("module_mynet_xyz_test2/predict2_" + str(seed) + ".txt")
result = np.loadtxt("module_mynet_xyz_test2/result2_" + str(seed) + ".txt")
# result = predict

temp1 = target[:, 0] - np.mean(target, axis=0)[0]
temp2 = result[:, 0] - np.mean(result, axis=0)[0]
fenzi = np.sum(temp1 * temp2)
fenmu = np.sqrt(np.sum(np.square(temp1))) * np.sqrt(np.sum(np.square(temp2)))
r = np.square(fenzi / fenmu)
    
plt.figure(figsize=(10, 7))
plt.plot(predict[:, 0], result[:, 0], "k*", markersize=12)
plt.plot(predict[:, 0], predict[:, 0], "r", linewidth=4)
plt.xlabel("True x", fontsize=50, fontweight="light")
plt.ylabel("Predict x", fontsize=50, fontweight="light")
plt.xticks([0.1, 0.3, 0.5], fontsize=40, fontweight="light")
plt.tick_params(direction="in", width=2, size=6)
plt.yticks([0.3, 0.5], fontsize=40, fontweight="light")
ax = plt.gca()
ax.spines["bottom"].set_linewidth(4)
ax.spines["top"].set_linewidth(4)
ax.spines["left"].set_linewidth(4)
ax.spines["right"].set_linewidth(4)
plt.text(0.2, 0.55, "$R^2={:.3f}$".format(r), fontsize=50, fontweight="light")
plt.show()
    
temp1 = target[:, 1] - np.mean(target, axis=0)[1]
temp2 = result[:, 1] - np.mean(result, axis=0)[1]
fenzi = np.sum(temp1 * temp2)
fenmu = np.sqrt(np.sum(np.square(temp1))) * np.sqrt(np.sum(np.square(temp2)))
r = np.square(fenzi / fenmu)

plt.figure(figsize=(10, 7))
plt.plot(predict[:, 1], result[:, 1], "k*", markersize=12)
plt.plot(predict[:, 1], predict[:, 1], "r", linewidth=2.5)
plt.xlabel("True y", fontsize=35)
plt.ylabel("Predict y", fontsize=35)
plt.tick_params(direction="in", width=2, size=6)
plt.xticks([0.1, 0.3, 0.5], fontsize=30)
plt.yticks([0.3, 0.5], fontsize=30)
plt.text(0.21, 0.5, "$R^2={:.3f}$".format(r), fontsize=30)
plt.show()

print(np.mean(np.square(result - target)))

temp1 = target[:, 2] - np.mean(target, axis=0)[2]
temp2 = result[:, 2] - np.mean(result, axis=0)[2]
fenzi = np.sum(temp1 * temp2)
fenmu = np.sqrt(np.sum(np.square(temp1))) * np.sqrt(np.sum(np.square(temp2)))
r = np.square(fenzi / fenmu)

plt.figure(figsize=(10, 7))
plt.plot(predict[:, 2], result[:, 2], "k*", markersize=12)
plt.plot(predict[:, 2], predict[:, 2], "r", linewidth=2.5)
plt.xlabel("True y", fontsize=35)
plt.ylabel("Predict y", fontsize=35)
plt.tick_params(direction="in", width=2, size=6)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.text(0.1, 0.2, "$R^2={:.3f}$".format(r), fontsize=30)
plt.show()
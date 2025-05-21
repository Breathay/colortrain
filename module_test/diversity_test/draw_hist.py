# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd

plt.rcParams['font.sans-serif']=['Arial']

mae_mpsn = [0.0016845072345968947, 0.00151869716468, 0.00208437070434, 0.00186865254685, 0.00210195742272,
            0.00194083223372, 0.00226632829066, 0.0022022814736, 0.00158875054783, 0.00200231273969]

mae_mdn = 0.0468518169859

err_mpsn = np.max(mae_mpsn) - np.mean(mae_mpsn)
err_mdn = mae_mdn / 6
err_tandem = 0.0002
err_cgan = 0.0002
err_vae = 0.0014
err = [err_mpsn, err_tandem, err_cgan, err_vae]

y = [np.mean(mae_mpsn), 0.0043, 0.0069, 0.0074]

x = ["MPSN", "TN", "CGAN", "VAE"]

color = [np.array([212, 131, 177]) / 255, np.array([98, 183, 150]) / 255, 
          np.array([234, 188, 142]) / 255, np.array([107, 155, 183]) / 255]

plt.figure(figsize=(10, 7), dpi=300)
plt.bar(x, y, width=0.5, color=color)

plt.ylim(0, 0.013)

plt.plot([x[1], x[1]], [y[3] + err[3] + 0.0003, y[3] + err[3] + 0.0003 + 0.001], "k", linewidth=1.5)
plt.plot([x[3], x[3]], [y[3] + err[3] + 0.0003, y[3] + err[3] + 0.0003 + 0.001], "k", linewidth=1.5)
plt.plot([x[1], x[3]], [y[3] + err[3] + 0.0003 + 0.001, y[3] + err[3] + 0.0003 + 0.001], "k", linewidth=1.5)
plt.text(x[2], y[3] + err[3] + 0.0021, "MAE from literatures", fontsize=25,
          horizontalalignment="center")

plt.errorbar(x=x, y=y, yerr=err, color="black", capsize=20,
             linestyle="None",
             marker="s", markersize=10, mfc="black", mec="black")

# plt.yscale("log")

plt.xticks(fontsize=25)
plt.yticks(fontsize=25)

plt.ylabel("Mean Absolute Error (MAE)", fontsize=30, fontweight="light")
plt.tick_params(direction="in", width=2.2, size=6, right=True, top=True, pad=16)
ax = plt.gca()
ax.spines["bottom"].set_linewidth(2)
ax.spines["top"].set_linewidth(2)
ax.spines["left"].set_linewidth(2)
ax.spines["right"].set_linewidth(2)
plt.show()




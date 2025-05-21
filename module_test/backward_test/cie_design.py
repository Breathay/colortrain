# -*- coding: utf-8 -*-
import sys
homepath = "../../"
sys.path.append(homepath)
import numpy as np
import pandas as pd
from Module.ColorDesign import DesignColor
import torch
import matplotlib.pyplot as plt
from PIL import Image
from model_predict import read_data, load_forward_model, color_predict
from Module.ColorDesign import MPSNDesign, CGANDesign, VAEDesign, MDNDesign, TNNDesign
from Module.ColorDesign import Standard
#计算设备
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
#############################
plt.rcParams['font.sans-serif']=['Arial']

def getLine(point1, point2, N):
    x1, y1 = point1
    x2, y2 = point2
    
    a = (y1 - y2) / (x1 - x2)
    b = y1 - a * x1
    
    x = np.linspace(x1, x2, N)
    y = a * x + b
    z = 1 - x - y
    return np.c_[x, y, z]

def drawmap():
    img = Image.open(homepath + "cie.png")
    im = np.array(img)
    plt.figure(figsize=(9, 9))
    plt.imshow(im, extent=[0, 1, 0, 1])

def drawPoint(xyz, xyz1, mk=10):
    # 画点
    plt.plot(xyz[:, 0], xyz[:, 1], "o", color=117/255 * np.ones(3), markersize=mk, label="target")
    plt.plot(xyz1[:, 0], xyz1[:, 1], "*", color="k", markersize=mk, label="design")

def draw(X_pred, points):

    drawmap()
    drawPoint(points, X_pred)
    plt.legend(fontsize=30, frameon=False, loc="lower right")
    ax = plt.gca()
    ax.spines["bottom"].set_linewidth(3)
    ax.spines["top"].set_linewidth(3)
    ax.spines["left"].set_linewidth(3)
    ax.spines["right"].set_linewidth(3)
    plt.xlim(0, 0.9)
    plt.ylim(0, 0.9)
    plt.xticks([0, 0.2, 0.4, 0.6, 0.8], fontsize=30)
    plt.yticks([0.1, 0.3, 0.5, 0.7, 0.9], fontsize=30)
    plt.xlabel("CIE $x$", fontsize=35)
    plt.ylabel("CIE $y$", fontsize=35)
    plt.tick_params(direction="in", width=2, size=6)
    plt.show()

def draw_color(X_pred):
    
    plt.figure(figsize=(9, 9))
    drawmap()
    drawPoint(points, X_pred, 15)
    # plt.legend(fontsize=30, frameon=False)
    ax = plt.gca()
    ax.spines["bottom"].set_linewidth(3)
    ax.spines["top"].set_linewidth(3)
    ax.spines["left"].set_linewidth(3)
    ax.spines["right"].set_linewidth(3)
    plt.axis("equal")

    # plt.xticks([0.1, 0.2, 0.3, 0.4, 0.5], fontsize=30)
    # plt.yticks([0.15, 0.25, 0.35, 0.45, 0.55, 0.65], fontsize=30)
    plt.xlim(0.1, 0.6)
    plt.ylim(0.15, 0.6)

    plt.tick_params(direction="in", width=2, size=6, pad=20)
    plt.show()


def get_points():
    res = 0.02

    points_x = np.arange(0.25, 0.43, res)
    points_y = np.arange(0.3, 0.48, res)

    X, Y = np.meshgrid(points_x, points_y)
    
    points = np.c_[X.reshape(-1, 1), Y.reshape(-1, 1)]
    return points

###########################################
X_test, s_test, mean_X, sd_X, mean_s, sd_s, index_SiH, index_SiO = read_data()
forward_state = load_forward_model()

mpsn = MPSNDesign(mean_X, sd_X, mean_s, sd_s)
cgan = CGANDesign(mean_X, sd_X, mean_s, sd_s)
vae = VAEDesign(mean_X, sd_X, mean_s, sd_s)
mdn = MDNDesign(mean_X, sd_X, mean_s, sd_s)
tnn = TNNDesign(mean_X, sd_X, mean_s, sd_s)
###########################################

points = get_points()
mpsn_s_pred, mpsn_X_pred = color_predict(mpsn, Standard(points, mean_X, sd_X), index_SiH, index_SiO, forward_state)
cgan_s_pred, cgan_X_pred = color_predict(cgan, Standard(points, mean_X, sd_X), index_SiH, index_SiO, forward_state)
vae_s_pred, vae_X_pred = color_predict(vae, Standard(points, mean_X, sd_X), index_SiH, index_SiO, forward_state)
mdn_s_pred, mdn_X_pred = color_predict(mdn, Standard(points, mean_X, sd_X), index_SiH, index_SiO, forward_state)
tnn_s_pred, tnn_X_pred = color_predict(tnn, Standard(points, mean_X, sd_X), index_SiH, index_SiO, forward_state)


draw_color(mpsn_X_pred)
draw_color(cgan_X_pred)
draw_color(vae_X_pred)
draw_color(mdn_X_pred)
draw_color(tnn_X_pred)


















# -*- coding: utf-8 -*-
import sys
homepath = "../../"
sys.path.append(homepath)
import matplotlib.pyplot as plt
from Module.ColorDesign import MPSNDesign, CGANDesign, VAEDesign, MDNDesign, TNNDesign
import numpy as np
import pandas as pd
import torch
from module_test.backward_test.model_predict import color_predict, load_forward_model, read_data, Standard

# 计算设备
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

if __name__ == "__main__":
    #############################
    X_test, s_test, mean_X, sd_X, mean_s, sd_s, index_SiH, index_SiO = read_data()
    forward_state = load_forward_model()

    mpsn = MPSNDesign(mean_X, sd_X, mean_s, sd_s)
    cgan = CGANDesign(mean_X, sd_X, mean_s, sd_s)
    vae = VAEDesign(mean_X, sd_X, mean_s, sd_s)
    mdn = MDNDesign(mean_X, sd_X, mean_s, sd_s)
    tnn = TNNDesign(mean_X, sd_X, mean_s, sd_s)
    #############################
    S_mpsn = []
    S_cgan = []
    S_vae = []
    S_mdn = []
    S_tnn = []
    #########################
    color = np.random.rand(1, 2) * 0.5
    for i in range(200):
        mpsn_s_pred, mpsn_X_pred = color_predict(mpsn, Standard(color, mean_X, sd_X), index_SiH, index_SiO, forward_state)
        cgan_s_pred, cgan_X_pred = color_predict(cgan, Standard(color, mean_X, sd_X), index_SiH, index_SiO, forward_state)
        vae_s_pred, vae_X_pred = color_predict(vae, Standard(color, mean_X, sd_X), index_SiH, index_SiO, forward_state)
        mdn_s_pred, mdn_X_pred = color_predict(mdn, Standard(color, mean_X, sd_X), index_SiH, index_SiO, forward_state)
        tnn_s_pred, tnn_X_pred = color_predict(tnn, Standard(color, mean_X, sd_X), index_SiH, index_SiO, forward_state)


        S_mpsn.append(mpsn_s_pred[0])
        S_cgan.append(cgan_s_pred[0])
        S_vae.append(vae_s_pred[0])
        S_mdn.append(mdn_s_pred[0])
        S_tnn.append(tnn_s_pred[0])

    S_mpsn = np.array(S_mpsn)
    S_cgan = np.array(S_cgan)
    S_vae = np.array(S_vae)
    S_mdn = np.array(S_mdn)
    S_tnn = np.array(S_tnn)


    np.savetxt("S_mpsn.txt", S_mpsn)
    np.savetxt("S_cgan.txt", S_cgan)
    np.savetxt("S_vae.txt", S_vae)
    np.savetxt("S_mdn.txt", S_mdn)
    np.savetxt("S_tnn.txt", S_tnn)

    #############################################

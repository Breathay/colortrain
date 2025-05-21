# -*- coding: utf-8 -*-
import colour
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from colour import SDS_ILLUMINANTS, SpectralDistribution
from colour.colorimetry import sd_to_XYZ



def getColor(R):
    l = np.linspace(400, 750, len(R))
    cs = CubicSpline(l, R)
    lamda = np.arange(400, 750, 5)
    data = cs(lamda)
    
    d = dict(np.c_[lamda, data])
    
    sd = SpectralDistribution(d)
    illuminant = SDS_ILLUMINANTS['D65']
    X, Y, Z = sd_to_XYZ(sd, illuminant=illuminant)
    # x, y = X / (X + Y + Z), Y / (X + Y + Z)
    rgb = colour.convert(sd, 'Spectral Distribution', 'sRGB', verbose={'mode': 'Short'},
                         sd_to_XYZ={'illuminant': illuminant})
    for i in range(len(rgb)):
        if rgb[i] < 0:
            rgb[i] = 0
        if rgb[i] > 1:
            rgb[i] = 1
    # if np.any(rgb > 1):
    #     rgb /= np.max(rgb)
    return rgb, (X, Y, Z)

if __name__ == "__main__":
    data = np.array(pd.read_csv("../data/dataset/data_spectrum_new.csv", index_col=0))
    
    ref = data[:, :50]
    s = data[:, 50:]
    data_color = []
    for i in range(len(ref)):
        rgb, (X, Y, Z) = getColor(ref[i])
        data_color.append(np.r_[rgb, X, Y, Z, s[i]])
        print(X, Y, Z)
        print("{} data finished".format(i))
    data_color = np.array(data_color)
    df = pd.DataFrame(data_color)
    df.to_csv("../data/dataset/data_new.csv")

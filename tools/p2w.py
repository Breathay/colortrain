# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

data = np.array(pd.read_csv("../data/dataset/data_new.csv", index_col=0))

color = data[:, :6]
s = data[:, 6:]

for i in range(len(s)):
    p, w, ww, d = s[i]
    
    w1 = (p - w) / 2
    w2 = d
    w3 = w / 2 - ww / 2 - d
    w4 = ww / 2
    
    data[i, 6:] = np.array([w1, w2, w3, w4])
    
    
df = pd.DataFrame(data)
df.to_csv("../data/dataset/data_w_new.csv")

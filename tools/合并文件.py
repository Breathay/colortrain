# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np


df00 = np.array(pd.read_csv("../data/dataset/data_spectrum_new.csv", index_col=0))
df10 = np.array(pd.read_csv("../data/dataset/data_spectrum4.csv", index_col=0))

data = np.r_[df00, df10]
df0 = pd.DataFrame(data)
df0.to_csv("../data/dataset/data_spectrum_new.csv")



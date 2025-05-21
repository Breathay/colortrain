# -*- coding: utf-8 -*-
import matlab
import matlab.engine
import numpy as np
import pandas as pd

def generate_structures(min_total, max_total, min_each):
    # 验证参数有效性
    sum_min = 4 * min_each
    if min_total < sum_min:
        raise ValueError("min_total must not be less than 4 times min_each")
    if max_total < min_total:
        raise ValueError("max_total must be greater than or equal to min_total")
    if sum_min > max_total:
        raise ValueError("The sum of minimum values exceeds max_total")
    
    # 随机生成总和S
    S = np.random.uniform(min_total, max_total)
    remaining = S - sum_min
    
    # 如果remaining为0，直接返回四个min_each
    if remaining == 0:
        return [min_each] * 4
    
    # 生成三个分割点并排序
    
    points = np.sort(np.random.uniform(0, remaining, 3))
    
    a, b, c = points
    
    # 计算四个增量部分
    deltas = np.array([a, b - a, c - b, remaining - c])
    
    # 打乱顺序以确保均匀性
    np.random.shuffle(deltas)
    
    # 生成最终数值
    numbers = min_each + deltas
    
    return numbers


def getSpectral(s, index_SiH, index_SiO, engine):
    
    w1, w2, w3, w4 = s
    p = (w1 + w2 + w3 + w4) * 2
    w = p - 2 * w1;
    ww = 2 * w4;
    d = w2;
    R = engine.eff_eval(matlab.double([p]), 
                        matlab.double([w]), 
                        matlab.double([ww]), 
                        matlab.double([d]), 
                        matlab.double(index_SiH.tolist()),
                        matlab.double(index_SiO.tolist()),
                        matlab.double([1]))
    R = np.array(R)[0]
    return R

if __name__ == "__main__":
    engine = matlab.engine.start_matlab()
    engine.addpath(engine.genpath('../reticolo_allege_v9'))
    
    index_SiH = np.array(pd.read_csv("index/index_SiH.csv"))
    index_SiO = np.array(pd.read_csv("index/index_SiO.csv"))
    
    epochs = 2000
    data = np.zeros((epochs, 54))
    
    n0 = 1
    
    for epoch in range(epochs):
    
        w1, w2, w3, w4 = generate_structures(200, 600, 20)
        p = (w1 + w2 + w3 + w4) * 2
        w = p - 2 * w1;
        ww = 2 * w4;
        d = w2;
        
        R = getSpectral((w1, w2, w3, w4), index_SiH, index_SiO, engine)
        
        cache = [p, w, ww, d]
        
        result = np.r_[R, cache]
        data[epoch, :] = result
        print("{} data calculated!".format(epoch))
        
    df = pd.DataFrame(data)
    df.to_csv("data_spectrum3.csv")
    
    from MileServer import MileServer
    server = MileServer()
    server.setEmileContent('Data Establishing Finished', 'calculate finished')
    server.sendEmile()






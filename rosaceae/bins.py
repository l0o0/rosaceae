# -*- coding: UTF-8 -*-
"""
rosaceae.bin
~~~~~~~~~~~~

This module implements data binning.
"""


import numpy as np
import pandas as pd



def bin_var(xarray, border=None):
    '''
    -xarray : a numpy array
    -border : a border list
    '''
    # 创建8个分区间
    if not border:
        des = xarray.describe()
        print des
        if des['75%'] < 7:
            step = (des['75%']-des['25%'])/3
        else:
            step = des['std']/2
            step = int(step)
        step = np.round(step, 3)
        border = [des['50%']+(i-3)*step for i in range(6)]
        border = [i for i in border if i >=0]
        #print 'old border: %s' % border
        if len(border) != 6:
            added = [border[-1]+ i*step for i in range(1,6-len(border))]
            border.extend(added)
        print 'border:%s, step: %s' % (border, step)
    else:
        print 'border:%s, step: Set' % (border, )
    out = {}
    for i, j in enumerate(border):
        if i == 0:
            k = '-inf,%s' % j
            tmp = np.where(np.logical_and(xarray>=0, xarray<j))[0]
        else:
            k = '%s,%s' % (border[i-1],j)
            tmp = np.where(np.logical_and(xarray>=border[i-1], xarray<border[i]))[0]
        out[k] = tmp
        print i,j, k
    out['%s,inf' % j] = np.where(xarray>=border[-1])[0]

    return out


# 根据数据的分位值来分箱
def bin_quantile(xarray, border=None):
    if len(xarray.unique()) < 7:
        border = xarray.unique().tolist()
        border.sort()
    else:
        border = [np.percentile(xarray, 0.05),
                  np.percentile(xarray, 0.2), np.percentile(xarray, 0.5),
                  np.percentile(xarray, 0.8), np.percentile(xarray, 0.95),
                ]
    #print border

    out = {}
    for i, j in enumerate(border):
        if i == 0:
            k = '-inf,%s' % j
            tmp = np.where(np.logical_and(xarray>=0, xarray<j))[0]
        else:
            k = '%s,%s' % (border[i-1],j)
            tmp = np.where(np.logical_and(xarray>=border[i-1], xarray<border[i]))[0]
        out[k] = tmp
    out['%s,inf' % j] = np.where(xarray>=border[-1])[0]
    return out



def bin_scatter(xarray, border = None):
    '''
    '''
    out = {}
    if not border:
        values = list(set(xarray))
        border = sorted(values)
    for i in border:
        out[i] = np.where(xarray == i)[0]
    return out

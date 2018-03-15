# -*- coding: UTF-8 -*-
"""
rosaceae.bin
~~~~~~~~~~~~

This module implements data binning.
"""


import numpy as np
import pandas as pd


def bin_frequency(xarray, bins=5):
    '''Data discretization by the same frequency.
    Data are binned by the same frequency. Frequency is controlled by bins
    number, frequency = (total length of xarray) / (bins number) .

    Args:
        xarray: Pandas.Series or Numpy.array type data.
        bins: int, bins number.
    '''

    xarray_sorted = sorted(xarray)
    step = len(xarray) / bins

    out = {}
    for i in range(bins):
        left = xarray_sorted[i*step]
        right = xarray_sorted[(i+1)*step]
        key = "%s,%s" % (left, right)
        out[key] = np.where(np.logical_and(xarray>=left, xarray<right))[0]

    return out


def bin_distance(xarray, bins=5):
    '''Data discretization in the same distance.
    Data are binned by the same distance, the distance is controlled by max
    interval and bins number. Max interval = max(xarray) - min(xarray),
    distance = (max interval) / (bins number).

    Args:
        xarray: Pandas.Series or Numpy.array type data.
        bins: int, bins number.
    '''
    distance = (max(xarray) - min(xarray)) / bins
    MIN = min(xarray)

    out = {}
    for i in range(bins):
        left = MIN + i * distance
        right = MIN + (i+1) * distance
        key = "%s,%s" % (left, right)
        out[key] = np.where(np.logical_and(xarray>=left, xarray<right))[0]

    return out


def bin_custom(xarray, border):
    '''Binning data by customize boundary.

    Args:
        xarray : a numpy array.
        border : a border list.
    '''
    out = {}

    for i, j in enumerate(border):
        k = '%s,%s' % (border[i-1],j)
        tmp = np.where(np.logical_and(xarray>=border[i-1], xarray<border[i]))[0]
        out[k] = tmp

    return out


def bin_scatter(xarray, border = None):
    '''Binning discretization data.
    '''
    out = {}
    if not border:
        values = list(set(xarray))
        border = sorted(values)
    for i in border:
        out[i] = np.where(xarray == i)[0]
    return out

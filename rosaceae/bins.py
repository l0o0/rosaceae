# -*- coding: UTF-8 -*-
"""
rosaceae.bin
~~~~~~~~~~~~

This module implements data binning.
"""

from __future__ import print_function

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier


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


def bin_scatter(xarray, border = None, na_omit=True):
    '''Binning discretization data.

    Parameteres
    -----------
    xarray : array like
        Input discretization data array. Data value is not continous numberic.
    border : array like
        Unfinished
    
    Returns
    -------
    Dictionary
        Category as key names. Corresponding row index list as values.
    '''
    out = {}
    if border is None:
        values = list(set(xarray[~pd.isnull(xarray)]))
        border = sorted(values)

    if not na_omit and sum(pd.isnull(xarray)) > 0:
        out['NA'] = np.where(pd.isnull(xarray))[0]
    for i in border:
        if i == 'None' or i == 'nan':
            continue 
        else:
            out[i] = np.where(xarray == i)[0]
    return out


def bin_tree(xarray, y, min_samples_node=0.05, na_omit=True, **kwargs):
    '''Binning data according DecisionTree node.

    Parameters
    ----------
    xarray : array like or ndarray
        Input feature value, array like, shape = [n_samples] or
        [n_samples, 1]
    y : array like
        The target value, array like, shape = [n_samples] or 
        [n_samples, n_output]
    **kwargs : **kwargs
        Keyword arguments for sklearn DecisionTree.
    
    Returns
    -------
    Dictionary
        Bin interval as key names. Corresponding row index list as values.
    '''
    n_samples = xarray.shape[0]
    #print n_samples
    clf = DecisionTreeClassifier(random_state=0, 
                                criterion='entropy',
                                min_samples_split=0.2,
                                max_leaf_nodes=6, 
                                min_impurity_decrease=0.001,
                                **kwargs)
                                  
    if len(xarray.shape) == 1:                                
        xarray = pd.DataFrame(xarray.values.reshape(n_samples, 1))

    if not na_omit:
        na_where = np.where(pd.isna(xarray.iloc[:,0]))[0]
    
    # reset index for x and y in case of IndexingError
    xarray.reset_index(drop=True, inplace=True)
    y.reset_index(drop=True, inplace=True)
    
    y = y[~pd.isna(xarray.iloc[:,0])]
    xarray_substitute = xarray.dropna()     # remove NA value for sklearn
                
    clf.fit(xarray_substitute, y)                                
    children_left = clf.tree_.children_left
    children_right = clf.tree_.children_right
    threshold = clf.tree_.threshold
    nodes_info = clf.tree_.__getstate__()['nodes']
    small_nodes = [i for i,j in enumerate(nodes_info) 
                    if j[-2] < n_samples * min_samples_node]

    # find leaf node in tree and get threshold in the leaf
    breaks = []

    for i,(l,r) in enumerate(zip(children_left, children_right)):
        if l != r and l not in small_nodes and r not in small_nodes:
            breaks.append(threshold[i])
    breaks.sort()
    breaks = [-np.inf] + breaks + [np.inf]

    out = {}
    for i, b in enumerate(breaks[1:]):
        start = breaks[i]
        end = b
        key = "%s:%s" % (start, end)
        out[key] = np.where((xarray >= start) & (xarray < end))[0]
    
    if not na_omit and len(na_where) > 0:
        out['NA'] = na_where
    
    return out


# For chi-square binning
def bin_chi2(xarray, label, bins_num, na_omit=True):
    '''Binning data by chi-square.

    Parameters
    ----------
    xarray:
    label:
    bins_num:
    na_omit:

    Returns
    -------
    '''

    return 

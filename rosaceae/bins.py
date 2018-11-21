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


def bin_frequency(xarray, bins=5, na_omit=False, verbose=False):
    '''Data discretization by the same frequency.

    Data are binned by the same frequency. Frequency is controlled by bins
    number, frequency = (total length of xarray) / (bins number). Data will
    be sorted in ascending. Missing values will be palced at the end.

    Parameters
    ----------
    xarray: Pandas.Series or Numpy.array type data.
    bins: int,
        Number of bins.
    na_omit: False or True
        Keep or drop missing value. Default is False, missing value will be grouped 
        in a separate bin.
    verbose: True or False, default is False
    
    Returns
    -------
    Dictionary
        Bin as key names. Corresponding row index as values.
    '''
    
    xarray = xarray.copy()
    xarray.reset_index(drop=True, inplace=True)
    xarray.sort_values(inplace=True)
    out = {}

    if na_omit:
        xarray = xarray[~pd.isna(xarray)]
    elif not na_omit and sum(pd.isna(xarray)) > 0:
        out['Miss'] = np.where(pd.isna(xarray))[0]

    step = int(len(xarray) / bins)
    if verbose:
        print('Step: %s'% step)

    for i in range(bins):
        group = 'Freq%s' % (i+1)
        if i == bins -1:
            out[group] = xarray.index[i*step:]
        else:
            out[group] = xarray.index[i*step:(i+1)*step]
    return out


def bin_distance(xarray, bins=5, na_omit=False, verbose=False):
    '''Data discretization in the same distance.

    Data are binned by the same distance, the distance is controlled by max
    interval and bins number. Max interval = max(xarray) - min(xarray),
    distance = (max interval) / (bins number).

    Parameters
    ----------
    xarray: Pandas.Series or Numpy.array 
    bins: Int
        Number of bins.
    na_omit: False or True
        Keep or drop missing value. Default is False, missing value will be grouped 
        in a separate bin.
    verbose: True or False, default is False

    Returns
    -------
    Dictionary
        Bin as key names. Corresponding row index as values.
    '''
    distance = (max(xarray) - min(xarray)) / bins

    xarray = xarray.copy()
    xarray.reset_index(drop=True, inplace=True)

    if verbose:
        print('Distance: %s' % distance)

    if not na_omit and sum(pd.isna(xarray)) > 0:
        out['Miss'] = np.where(pd.isna(xarray))[0]

    MIN = min(xarray)
    out = {}
    for i in range(bins):
        if i ==0:
            left = -np.inf
        else:
            left = MIN + i * distance

        if i == bins-1:
            right = np.inf 
        else:
            right = MIN + (i+1) * distance
        key = "[%s,%s)" % (left, right)
        out[key] = xarray.index[(xarray>=left) & (xarray<right)]

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
        out['Miss'] = np.where(pd.isnull(xarray))[0]
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
        key = "[%s:%s)" % (start, end)
        out[key] = np.where((xarray >= start) & (xarray < end))[0]
    
    if not na_omit and len(na_where) > 0:
        out['Miss'] = na_where
    
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


# For custom binning. 
def bin_custom(xarray, groups, na_omit=False, verbose=False):
    '''Binning data by customized binning boundary

    Parameters
    ----------
    xarray: array like data

    groups: list like 
        Custom binning boundary. For numeric data, ['(-inf:3]', '(3:6]', '(6:inf)'].
        For categorious data, [('A', 'B'), ('C'), ('D','E', 'Miss')], `Miss` is used 
        for missing data(NA values). However you can not put `Miss` in numeric data.
    na_omit: True or False
        Default is False. Get all NA in a separate group. If `Miss` in a customized 
        categorious data, this Miss separate group will delete from output.
    verbose: True or False, 
        Default is False. Print verbose message.
    '''
    # reset index 
    xarray = xarray.copy()
    xarray.reset_index(drop=True, inplace=True)

    if not isinstance(groups, list):
        raise TypeError('groups is a list, contains customized binning boundary')

    out = {}

    # handle missing data.
    if not na_omit:
        if verbose:
                print('Keep NA data')
        tmp = np.where(pd.isna(xarray))[0]
        if verbose:
                print('Missing data: %s' % len(tmp))
        if len(tmp) > 0:
            out['Miss'] = np.where(pd.isna(xarray))[0]

    if ':' in groups[0]:    # numeric custom

        for g in groups:
            if verbose:
                print('* Handling %s' % g)
            start, end = pd.to_numeric(g[1:-1].split(':'))
            if g.startswith('(') and g.endswith(']'):
                out[g] = np.where((xarray > start) & (xarray <= end))[0]
            elif g.startswith('[') and g.endswith(']'):
                out[g] = np.where((xarray >= start) & (xarray <= end))[0]
            elif g.startswith('(') and g.endswith(')'):
                out[g] = np.where((xarray > start) & (xarray < end))[0]
            elif g.startswith('[') and g.endswith(')'):
                out[g] = np.where((xarray >= start) & (xarray < end))[0]
    else:   # category custom
        for g in groups:
            if verbose:
                print('* Handling %s' % str(g))
            if isinstance(g, str):
                g = (g,)
            if 'Miss' in g:
                out[g] = np.where((xarray.isin(list(g))) | (pd.isna(xarray)))[0]
                del out['Miss']
            else:
                out[g] = np.where(xarray.isin(list(g)))[0]
    
    return out
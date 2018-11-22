# -*- coding: UTF-8 -*-
"""
rosaceae.bin
~~~~~~~~~~~~

This module implements data binning.
"""

from __future__ import print_function

import numpy as np
import pandas as pd
from itertools import combinations
from sklearn.tree import DecisionTreeClassifier


def bin_frequency(xarray, bins=5, na_omit=True, verbose=False):
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
        Keep or drop missing value. Default is True, missing value will be grouped 
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


def bin_distance(xarray, bins=5, na_omit=True, verbose=False):
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
        Keep or drop missing value. Default is True, missing value will be grouped 
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



def bin_tree(xarray, y, min_samples_node=0.05, na_omit=True, **kwargs):
    '''Binning data according DecisionTree node.

    Parameters
    ----------
    xarray : pandas series data
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
def chi2(a, bad_rate):
        b = [int(sum(a)*bad_rate), sum(a)- int(sum(a)*bad_rate)]
        chi = (a[0] - b[0])**2 / b[0] + (a[1] - b[1])**2 / b[1]
        return chi

    
def recursion(groups, counts, bins, numeric=False, verbose=False):
    max_chi = 0
    if not numeric:
        for _i, i in combinations(range(len(groups)), 2):
            com = (_i, i)
            com_count = counts[i] +  counts[_i]
            tmpchi = chi2(com_count, bad_rate)
            if tmpchi > max_chi:
                max_chi = tmpchi
                max_com_idx = com
        # merge similar categories into one 
        if verbose:
            print('("{0}") + ("{1}") --> ("{0},{1}")'.format(groups[max_com_idx[0]], groups[max_com_idx[1]]))
        merged = '%s,%s' % (groups[max_com_idx[0]], groups[max_com_idx[1]])
        groups = [g for _, g in enumerate(groups) if _ not in max_com_idx]
        merged_counts = counts[max_com_idx[0]] + counts[max_com_idx[1]]
        counts = [c for _, c in enumerate(counts) if _ not in max_com_idx]
        groups.append(merged)
        counts.append(merged_counts)
    else:
        max_com_idx = tuple()
        for i in range(1, len(groups)-1):
            chi_before = chi2(counts[i-1] + counts[i], bad_rate)
            chi_after = chi2(counts[i] + counts[i+1], bad_rate)
            if chi_before > max_chi:
                max_com_idx = (i-1, i)
                max_chi = chi_before
            elif chi_after > max_chi:
                max_com_idx = (i, i+1)
                max_chi = chi_after
        merged = (groups[max_com_idx[0]][0], groups[max_com_idx[1]][1]) # create a new boundary
        if verbose:
            print(groups[max_com_idx[0]], groups[max_com_idx[1]], '-->' ,merged)
        groups = groups[:max_com_idx[0]] + [merged] + groups[max_com_idx[1]+1:]
        merged_counts = counts[max_com_idx[0]] + counts[max_com_idx[1]]
        counts = counts[:max_com_idx[0]] + [merged_counts] + counts[max_com_idx[1]+1:]
    if len(groups) <= bins:
        return groups
    else:
        return recursion(groups, counts, bins, numeric=numeric, verbose=verbose)

        
def bin_chi2(xarray, y, bins, min_sample=0.01, na_omit=True, verbose=False):
    xarray = xarray.copy()
    xarray.reset_index(drop=True, inplace=True) 
    out = {}
    # remove missing values or not 
    if na_omit:
        xarray = xarray[~pd.isna(xarray)]
        y = y[~pd.isna(xarray)]        
    elif not na_omit:
        out['Miss'] = xarray.index[pd.isna(xarray)]
    total_bad = xarray.sum()
    bad_rate = total_bad / len(y)
    # numeric or categorious
    if xarray.dtype == 'object':
        if verbose:
            print('Categorious data detected.')
        groups = list(set(xarray[~pd.isna(xarray)]))
        counts = []
        for g in groups:
            tmp = y[xarray==g]
            counts.append(np.array(sum(tmp), len(tmp)-sum(tmp)))
        groups = recursion(groups, counts, bins, numeric=False, verbose=verbose)
    else:
        if verbose:
            print('Numeric data detected.')
        rounds = 50
        q25 = np.percentile(xarray, 25)
        q75 = np.percentile(xarray, 75)
        iqr = q75-q25 
        min_value = max(min(xarray), q25 - 1.5*iqr)
        max_value = min(max(xarray), q75 + 1.5*iqr)
        step = (max_value - min_value) / rounds # exclude outlier
        borders = [-np.inf] + [min_value + step *i for i in range(1,rounds)] + [np.inf]
        
        if verbose:
            print("Range from min value: %s, max value: %s, step: %s" % (min_value, max_value, step))    
        groups = []
        counts = []
        
        # create a boundary list and remove small group
        i_start = 0
        for _i, b in enumerate(borders[1:]):
            start = borders[i_start]
            end = b
            if sum((xarray >= start) & (xarray < end)) < len(y) * min_sample:
                continue
            else:
                tmp = y[(xarray >= start) & (xarray < end)]
                groups.append((start, end))
                counts.append(np.array((sum(tmp), len(tmp)-sum(tmp))))
            i_start = _i + 1
        # add +inf if deleted 
        if groups[-1][1] != np.inf:
            start = groups[-1][0]
            end = np.inf
            tmp = y[(xarray >= start) & (y < end)]
            groups[-1] = (start, end)
            counts[-1] = np.array((sum(tmp), len(tmp)-sum(tmp)))
            
        if verbose:
            print('Init groups:', groups)
        groups = recursion(groups, counts, bins, numeric=True, verbose=verbose)

    return groups


# For custom binning. 
def bin_custom(xarray, groups, na_omit=True, verbose=False):
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

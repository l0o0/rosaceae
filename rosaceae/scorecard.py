# -*- coding: utf-8 -*-
'''
rosaceae.scorecard
~~~~~~~~~~~~~~~~~~

This module provides functions for credit risk scorecard.
'''
from __future__ import print_function
import pandas as pd

from math import log, e

from .bins import bin_tree


# 计算 woe 值
def getWOE(x, y, bins_out=None, good_label=0, verbose=False):
    '''Calculate WOE value.
    WOE(weight of evidence)
    1 indicates good case, 0 indicates bad case.

    Args:
        -x : array like.
        -y : array like, target value.

    Returns:
    '''
    total_good = float(sum(y==good_label))
    total_bad = y.shape[0] - total_good
    out = {}

    if bins_out == None:
        if verbose:
            print("Binning data by rosaceae.bins.bin_tree")
        bins_out = bin_tree(x, y)

    for border in bins_out:
        border_good = float(sum(y[bins_out[border]]==good_label))
        border_bad = len(bins_out[border]) - border_good
        border_woe = log((border_bad/total_bad)/(border_good/total_good))
        out[border] = border_woe        
    return out


def getWOE_a(vars, y, data, good_label=0, verbose=False, table=False):
    '''A wrapper of getWOE
    Calculate woe for var in vars. Vars should be in data's columns.
    A dict with var and its output from getWOE is returned.
    '''
    out = {}
    for v in vars:
        v_out = getWOE(data[v], data[y], good_label=good_label, verbose=verbose)
        out[v] = v_out
    if table:
        df = pd.DataFrame(columns=['Feature', 'Bin', 'WOE'])
        for v in out:
            for b in out[v]:
                df = df.append({'Feature':v, 'Bin':b, 'WOE':out[v][b]}, ignore_index=True)
        out = df
    return out


# raw value transfer to woe value according woe table
def woe_replace(woes, data):
    '''Use woe value to replace original value.

    '''
    if isinstance(woes, dict):
        cols = woes.keys()
    elif isinstance(woes, pd.DataFrame):
        cols = woes['Feature'].unique()

    woe_data = data.loc[:, data.columns.isin(cols)].copy()
    for var in woes:
        var_woe = woes[var]
        for border in var_woe:
            start, end = pd.to_numeric(border.split(':'))
            flags = ((woe_data[var]>= start) & (woe_data[var]<end))
            woe_data.loc[flags, var] = var_woe[border]
    return woe_data



# 计算 IV 值
def iv(x, y, good_label=0, verbose=False):
    '''Calculate feature iv value.
    Args:
        x: pandas series.
        y: target value. 
        bins_out: output of bins function.
        good_label: label for good case.
    '''
    bins_out = bin_tree(x, y)
    total_good = float(sum(y==good_label))
    total_bad = y.shape[0] - total_good
    if verbose:
        print("total_good: %s\ttotal_bad: %s\n" % (total_good, total_bad))
        print("Features\tBin\tGood(%)\tBad(%)\twoe_i\tiv_i") 
    iv = 0
    for border in bins_out:
        border_good = float(sum(y[bins_out[border]]==good_label)) 
        border_bad = len(bins_out[border]) - border_good
        border_woe = log((border_bad/total_bad)/(border_good/total_good))
        iv_i = (border_bad/total_bad - border_good/total_good) * border_woe
        if verbose:
            print("%s\t%s\t%s\t%s\t%s\t%s" % (
                    x.name,
                    border,
                    border_good/total_good * 100, 
                    border_bad/total_bad * 100,
                    border_woe, 
                    iv_i))
        iv += iv_i

    return iv


def getConstant(theta, pdo, basescore, data, woe_table, verbose=False):
    '''Calculata Shift and Slope
    The score of an individual i is given by the formula:

        Score(i) = Shift + Slope*(b0 + b1*WOE1(i) + b2*WOE2(i)+ ... +bp*WOEp(i))

    where bj is the coefficient of the j-th variable in the model,
    and WOEj(i) is the Weight of Evidence (WOE) value for the
    i-th individual corresponding to the j-th model variable.

    In short formula:

        Score = Shift + Slope*ln(Good/Bad)
        Score + PDO = Shift + Slope*ln(2* Good/Bad)

    where Slope = PDO / ln(2), Shift = Score - Slope*ln(Good/Bad).

    Args:
        theta: the ratio of Good/Bad. Let good ratio is p, then bad ratio is
            (1-p), theta = p/(1-p).
        pdo: Point-to-Double Odds. When the odds is doubled, score will increate pdo.
        basescore: When the ratio of Good/Bad is theta, the score is basescore.
    '''
    slope = pdo/log(2, e)
    shift = basescore - slope * log(float(theta), e)
    if verbose:
        print("Shift is %s, slope is %s" % (shift, slope))
    return (shift, slope)


def getScore(woe_table, xarray, missing=0):
    score = 0
    xarray.fillna(0, inplace=True)
    for idx in xarray.index:
        value = xarray[idx]
        
        if pd.isna(value):
            score += missing
            continue

        tmp_woe = woe_table.loc[woe_table['Feature'] == idx, :]
        for k in tmp_woe['Bin']:
            border = pd.to_numeric(k.split(':'))
            if value >= border[0] and value < border[1]:
                score += tmp_woe.loc[tmp_woe['Bin']==k, 'Score'].values[0]
                break
    return score

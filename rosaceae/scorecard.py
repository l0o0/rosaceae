# -*- coding: utf-8 -*-
'''
rosaceae.scorecard
~~~~~~~~~~~~~~~~~~

This module provides functions for credit risk scorecard.
'''
from __future__ import print_function
import pandas as pd

from math import log, e

from .bins import bin_tree, bin_scatter


# raw value transfer to woe value according woe table
def woe_replace(woes, data):
    '''Use woe value to replace original value.
    Args:
        -woe : woe value dictionary or pandas data frame.
        -data : raw data used to replace with woe value.
    '''
    if isinstance(woes, dict):
        cols = woes.keys()
    elif isinstance(woes, pd.DataFrame):
        if sum(woes.columns.isin(['Variable', 'Bin', 'WOE'])) != 3:
            raise ValueError('DataFrame should contains Variable, Bin and WOE')
        tmp = {}
        for i, row in woes.iteritems():
            var = row['Variable']
            border = row['Bin']
            woe = row['WOE']
            if var not in tmp:
                tmp[var] = {}
            tmp[var][border] = [woe]
        woes = tmp
    else:
        raise TypeError("WOE should be dictionary or pandas DataFrame.")

    woe_data = data.loc[:, data.columns.isin(cols)].copy()
    for var in woes:
        var_woe = woes[var]
        for border in var_woe:
            start, end = pd.to_numeric(border.split(':'))
            flags = ((woe_data[var]>= start) & (woe_data[var]<end))
            woe_data.loc[flags, var] = var_woe[border][0]
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


# 计算 IV 值
def woe_iv(data, y, vars=None, good_label=0, dt=None, min_samples_node=0.05, na_omit=True, 
           verbose=False, **kwargs):
    '''Calculate feature iv value.
    Args:
        data: pandas data frame.
        y: target value column name. 
        vars: variables, list like.
        good_label: label for good case, default is 0.
        dt: list like contains 0 and 1, the length is the same as the number of variables. 0 
        indicates numberic data, 1 indicates category data.
        na_omit: remove NAN value, default is True.
        verbose: print verbose information.
    Returns:
        A pandas data frame is returned. Contains variable, bin, good/bad case number 
        and corresponding woe and iv value.
    '''
    if len(set(data[y]))!= 2:
        raise TypeError('Need binary value in label.')

    bad_label = list(set(data[y]) - set([good_label]))[0]

    if vars == None:
        vars = data.columns.tolist()
        vars.remove(y)

    if dt == None or len(dt) != len(vars):
        raise TypeError("dt argument is a list contains 0 and 1, \
                    whose length equals variable's length.")

    info_df = pd.DataFrame(columns=['Variables', 'Bin', 'Good', 'Bad', 'pnt_%s'% good_label, 
                            'pnt_%s' % bad_label, 'woe', 'iv_i'])

    for idx, var in enumerate(vars):
        if verbose:
            print("Processing on %s" % var)

        if na_omit:
            total_good = float(sum(data[y][~pd.isna(data[var])] == good_label))
            total_bad = len(data[y][~pd.isna(data[var])]) - total_good
        else:
            total_good = float(sum(data[y]==good_label))
            total_bad = data.shape[0] - total_good

        if dt[idx] == 0:
            bins_out = bin_tree(data[var], data[y], na_omit=na_omit, min_samples_node= min_samples_node, **kwargs)
        elif dt[idx] == 1:
            bins_out = bin_scatter(data[var])
            
        #print(bins_out.keys())
        if verbose:
            print("total_good: %s\ttotal_bad: %s\n" % (total_good, total_bad))
            print("Variables\tBin\tGood(%)\tBad(%)\twoe_i\tiv_i") 
        
        for border in bins_out:
            border_good = float(sum(data[y][bins_out[border]]==good_label)) 
            border_bad = len(bins_out[border]) - border_good
            border_woe = log((border_bad/total_bad)/(border_good/total_good))
            iv_i = (border_bad/total_bad - border_good/total_good) * border_woe
            row = [var,
                    border,
                    border_good,
                    border_bad,
                    border_good/total_good , 
                    border_bad/total_bad,
                    border_woe, 
                    iv_i]
            info_df.loc[info_df.shape[0], :] = row
            if verbose:
                print('\t'.join([str(_i) for _i in row]))
    iv = info_df.groupby('Variables').iv_i.sum()
    info_df['iv'] = info_df['Variables'].map(lambda x:iv[x])
    return info_df


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

def getScoreCard(woe_table):


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

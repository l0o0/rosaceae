# -*- coding: utf-8 -*-
'''
rosaceae.scorecard
~~~~~~~~~~~~~~~~~~

This module provides functions for credit risk scorecard.
'''

from __future__ import print_function
import pandas as pd
import numpy as np

from math import log, e

from .bins import bin_tree, bin_scatter


# raw value transfer to woe value according woe table
def replaceWOE(woes, data):
    '''Use WOE value to replace original value.

    Parameters
    ----------
    woe : dictionary or pandas.DataFrame
        WOE value dictionary or pandas data frame.
    data : pandas.DataFrame
        Raw data used to replace with WOE value.

    Returns
    -------
    pandas.DataFrame
        A data frame is returned which has the sample shape with input data frame.
    '''
    if isinstance(woes, dict):
        cols = woes.keys()
    elif isinstance(woes, pd.DataFrame):
        if sum(woes.columns.isin(['Variable', 'Bin', 'WOE'])) != 3:
            raise ValueError('DataFrame should contains Variable, Bin and WOE')
        tmp = {}
        cols = woes['Variable'].unique()
        for i, row in woes.iterrows():
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
            if isinstance(border, str) and ':' in border:
                start, end = pd.to_numeric(border.split(':'))
                flags = ((woe_data[var]>= start) & (woe_data[var]<end))
            elif border == 'NA':
                flags = pd.isna(woe_data[var])
            else:
                flags = woe_data[var] == border
            woe_data.loc[flags, var] = var_woe[border][0]
    return woe_data


# information value and WOE
def woe_iv(data, y, vars=None, good_label=0, dt=None, min_samples_node=0.05, na_omit=True, 
           verbose=False, **kwargs):
    '''Calculate feature iv value.

    Parameters
    ----------
    data : pandas.DataFrame
        A data frame contians target label and features as columns.
    y : str
        Target or label column name.
    vars : list or array like
        A list contains column name, these columns can be viewd as  independent variables 
        or interested features. If `vars` is None, All columns are regarded as features except `y`.
    good_label : str or int
        Label for good case, default is 0.
    dt : list or array like
        The length is the same as the number of variables. 0 indicates numberic data, 
        1 indicates category data. 
    na_omit : bool
        Whether to remove NAN value, default is True.
    verbose : bool
        Print verbose information, default is False

    Returns
    -------
    pandas.DataFrame
        A pandas data frame is returned. Contains variable, bin, good/bad case number 
        and corresponding woe and iv value.
    '''
    var_y = data[y].copy()
    var_y.reset_index(drop=True, inplace=True)
    if len(set(var_y))!= 2:
        raise TypeError('Need binary value in label.')

    bad_label = list(set(var_y) - set([good_label]))[0]

    if vars is None:
        vars = data.columns.tolist()
        vars.remove(y)

    if dt == None or len(dt) != len(vars):
        raise TypeError("dt argument is a list contains 0 and 1, whose length equals variable's length.")

    info_df = pd.DataFrame(columns=['Variable', 'Bin', 'Good', 'Bad', 'pnt_%s'% good_label, 
                            'pnt_%s' % bad_label, 'WOE', 'IV_i'])

    for idx, var in enumerate(vars):
        if verbose:
            print("Processing on %s" % var)

        # reset index
        var_x = data[var].copy()
        var_x.reset_index(drop=True, inplace=True)

        if na_omit:
            total_good = float(sum(var_y[~pd.isna(var_x)] == good_label))
            total_bad = len(var_y[~pd.isna(var_x)]) - total_good
        else:
            total_good = float(sum(var_y==good_label))
            total_bad = data.shape[0] - total_good

        if dt[idx] == 0:
            bins_out = bin_tree(var_x, var_y, na_omit=na_omit, min_samples_node= min_samples_node, **kwargs)
        elif dt[idx] == 1:
            bins_out = bin_scatter(var_x, na_omit=na_omit)

        #print(bins_out.keys())
        if verbose:
            print("total_good: %s\ttotal_bad: %s\n" % (total_good, total_bad))
            print("Variable\tBin\tGood(%)\tBad(%)\tWOE_i\tIV_i") 
        
        for border in bins_out:
            border_good = float(sum(var_y[bins_out[border]]==good_label)) 
            border_bad = len(bins_out[border]) - border_good

            # In case of divide zero error, set border_good = 1 when it's 0
            if border_good == 0:
                border_good = 1
            if border_bad == 0:
                border_bad = 1

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
    iv = info_df.groupby('Variable').IV_i.sum()
    info_df['IV'] = info_df['Variable'].map(lambda x:iv[x])
    return info_df


def getScoreCard(woe_table, coef, inter, A, B):
    '''Contruct a score card table.

    According score card formula:

        Score(i) = A - B*(b0 + b1*WOE1(i) + b2*WOE2(i)+ ... +bp*WOEp(i))
    
    where bj is the coefficient of the j-th variable in the model,
    and WOEj(i) is the Weight of Evidence (WOE) value for the
    i-th individual corresponding to the j-th model variable.

    In short formula:

        Score = A - B*ln(Good/Bad)
        Score + PDO = A - B*ln(2* Good/Bad)

    where B = -PDO / ln(2), A = Score + B*ln(Good/Bad).

    A, B is needed for get score.

    Parameters
    ----------
    woe_table: pandas.DataFrame
        Output of function `woe_iv`.
    coef : dictionary
        Coefficient of the variables in the logistic regression, 
        variable(features) as key.
    inter : float 
        Interception of logistic regression.
    A : float
        Compensation points, calculated from score card formula above.
    B : float
        Scale factor, calculated from score card formula above.
    
    Returns
    -------
    pandas.DataFrame
        A score card data frame is returned. Each variable can get its own 
        score value in different interval. WOE and IV is also calculated for 
        each intervalu.
    '''
    scores = []
    for i, row in woe_table.iterrows():
        woe = row['WOE']
        var = row['Variable']
        if var not in coef:
            score = '--'
        else:
            score = - B * coef[var] * woe
        scores.append(score)
    basescore = A - B * inter
    scorecard = woe_table.copy()
    scorecard['Score'] = scores
    scorecard.loc[scorecard.shape[0]] = ['basescore'] + ['--'] * (scorecard.shape[1]-2) + [basescore]
    scorecard['ScoreRaw'] = scorecard['Score']
    scorecard['Score'] = scorecard['Score'].map(lambda x:x if x=='--' else int(np.round(x, 0)))
    scorecard['Bin'] = scorecard['Bin'].fillna('NA')
    scorecard.reset_index(drop=True, inplace=True)
    return scorecard

        
# Get case score
def getScore(data, scorecard, na_value=None):
    '''Calculate total score for case.

    Parameteres
    -----------
    data : pandas.DataFrame
        Input original data frame, variables as columns name.
    scorecard : pandas.DataFrame 
        Output of function `getScoreCard`.
    na_value : float
        Value for NaN, default is None.
    
    Returns
    -------
    pandas.Series
        According score card data frame, each row's final score in data 
        is calculated. Variables in data columns but not in score card 
        data frame will be ignored.
    '''
    # remove empty score row
    basescore = scorecard.loc[scorecard['Variable'] == 'basescore', 'Score'].astype('float').round(0)
    basescore = int(basescore)
    # create empty data frame whose shape is [data.shape[0], target_variable counts]
    cols = scorecard['Variable'].unique().tolist()
    cols.remove('basescore')
    tmpdata = pd.DataFrame(0, columns=cols, 
            index=np.arange(data.shape[0]))

    for i, row in scorecard.iterrows():
        score = row['Score']
        border = row['Bin']
        var = row['Variable']

        if var == 'basescore' or score == '--' or border == '-inf:inf':
            continue
        
        if pd.isna(border) or border == 'NA':            
            flags = pd.isna(data[var])
        elif ':' in border:            
            start, end = pd.to_numeric(border.split(':'))
            flags = ((data[var]>= start) & (data[var]<end))
        else:
            flags = (data[var] == border)
        tmpdata.loc[flags, var] = score
    return tmpdata.apply(sum, axis=1) + basescore
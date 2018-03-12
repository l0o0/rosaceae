# -*- coding: utf-8 -*-
'''
rosaceae.scorecard
~~~~~~~~~~~~~~~~~~

This module provides functions for credit risk scorecard.
'''

from math import log, e


def getWOE(c, y):
    '''Calculate WOE value.
    WOE(weight of evidence)
    1 indicates good case, 0 indicates bad case.

    Args:
        -c : dictionary, result of bin function.
        -y : pandas.Series or numpy.array, label.

    Returns:
    '''
    totalgood = np.count_nonzero(y)
    totalbad = len(y) - totalgood

    out = {}

    for k in c:
        region = y[c[k]]
        bad = np.count_nonzero(region)
        good = len(region) - bad
        #print len(region), good, bad
        if bad == 0 or good ==0:
            continue
        woe = log((float(bad)/b)/(float(good)/g))
        out[k] = woe
    return out


def get_constant(theta, pdo, basescore):
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
    shift = basescore - B * log(float(theta), e)
    return (shift, slope)


def getScore(woe_table, xarray):
    score = 0
    xarray.fillna(0, inplace=True)
    for idx in xarray.index[2:]:
        value = xarray[idx]
        tmp_woe = woe_table[idx]
        for k in tmp_woe:
            border = pd.to_numeric(k.split(':'))
            #print k, border
            if value >= border[0] and value < border[1]:
                #print idx, value, border, tmp_woe[k]
                score += tmp_woe[k]
                break
    return score

# -*- coding: utf-8 -*-
"""
rosaceae.utils
~~~~~~~~~~~~~~

This module provides utility functions that are used within Rosaceae.
Including visulization and summary functions.
"""
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from functools import reduce
from itertools import combinations
from math import log
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score


def model_selector(x_train, x_test, y_train, y_test, start=1, end=None, thresh=None, verbose=False):
    '''
    '''

    def factorial(m,n):
        return reduce(lambda x,y:x*y, range(1,m+1)[-n:])
    
    def com_count(m, n):
        return factorial(m,n)/factorial(n,0)


    result_df = pd.DataFrame(columns=['Var_No', 'Vars', 'train_score', 'test_score','coef', 'inter'])
    if not end:
        end = x_train.shape[1]
    cols = x_train.columns

    # total try counts
    total_try = sum([com_count(len(cols), n) for n in range(start, end+1)])
    print("Total events: %s" % total_try)

    for n in range(start, end+1):
        cols_try = combinations(cols, n)

        if not verbose:
            print(n)

        for t in cols_try:
            tmp_train = x_train.loc[:, x_train.columns.isin(t)]
            tmp_test = x_test.loc[:, x_test.columns.isin(t)]
            clf = LogisticRegression(random_state=0)
            clf.fit(tmp_train, y_train)
            train_roc_score = roc_auc_score(y_train, clf.decision_function(tmp_train))
            test_roc_score = roc_auc_score(y_test, clf.decision_function(tmp_test))
            
            if verbose:
                print("%s\t%s\t%s\t%s" % (n, ','.join(t), train_roc_score, test_roc_score))

            if thresh and test_roc_score < thresh:
                continue

            row = 0 if pd.isna(result_df.index.max()) else result_df.index.max() + 1
            result_df.loc[row] = [n, ','.join(t), train_roc_score, test_roc_score, clf.coef_[0], clf.intercept_[0]]

    return result_df


# Use cross validation to evaluate models.
def model_selector2(x, y, start=1, end=None, verbose=False):
    result_df = pd.DataFrame(columns=['Var_No', 'Vars', 'score', 'std'])
    if not end:
        end = x.shape[1]
    cols = x.columns
    for n in range(start, end+1):
        cols_try = combinations(cols, n)

        if not verbose:
            print(n)

        for t in cols_try:
            tmp = x.loc[:, x.columns.isin(t)]
            cv = StratifiedKFold(n_splits=10, shuffle=True)
            clf = LogisticRegression(random_state=0)
            scores = cross_val_score(clf, tmp, y)
            if verbose:
                print("%s\t%s\t%s\t%s" % (n, ','.join(t), scores.mean(), scores.std()))

            #row = 0 if pd.isna(result_df.index.max()) else result_df.index.max() + 1
            result_df.loc[result_df.shape[0]] = [n, ','.join(t), scores.mean(), scores.std()]

    return result_df


# Function for data eda.
def summary(data, verbose=False):
    '''Generates descriptive statistics for a dataset

    Generates descriptive statistics that summarize the central tendency,
    dispersion and shape of a dataset's distribution. When data is categorious 
    or datatime format, some statistics values will be replaced by NA. 

    Parameters
    ----------
    data : pandas data frame
    verbose : True or False
            Print verbose message, default is False.
    
    Returns
    -------
    A pandas data frame with statistics values for each column.
    Field: Field name.
    Type: object, numeric, integer, other.
    Recs: Number of records.
    Miss: Number of missing records.
    Min: Minimum value.
    Q25: First quartile. It splits off the lowest 25\% of data from the highest 75\%.
    Q50: Median or second quartile. It cuts data set in half.
    Avg: Average value.
    Q75: Third quartile. It splits off the lowest 75\% of data from the highest 25\%.
    Max: Maximum value.
    StDv: Standard deviation of a sample.
    Neg: Number of negative values.
    Pos: Number of positive values.
    OutLo: Number of outliers. Records below \code{Q25-1.5*IQR}, where \code{IQR=Q75-Q25}. 
    OutHi: Number of outliers. Records above \code{Q75+1.5*IQR}, where \code{IQR=Q75-Q25}.
    '''
    tmpdf = pd.DataFrame(columns=['Field', 'Type', 'Recs', 'Miss', 'Min', 'Q25', 'Q50', 
                                  'Avg', 'Q75', 'Max', 'StDv', 'Uniq', 'OutLo', 'OutHi'])
    for i,col in enumerate(data.columns):
        datatype = str(data[col].dtype)
        recs = len(data[col])
        miss = sum(pd.isna(data[col]))
        uniq = len(data[col].unique())
        if verbose:
            print(col)
        if datatype == 'object' or 'datetime' in datatype:
            _min = np.nan
            _max = np.nan
            q25 = np.nan
            q50 = np.nan
            avg = np.nan
            q75 = np.nan
            stdv = np.nan
            outlo = np.nan
            outhi = np.nan            
        else:
            desc = data[col].describe()
            _min = desc['min']
            _max = desc['max']
            q25 = float(format(desc['25%'], '.4g'))
            q50 = float(format(desc['50%'], '.4g'))
            q75 = float(format(desc['75%'], '.4g'))
            avg = float(format(desc['mean'], '.4g'))
            stdv = float(format(desc['std'], '.4g'))
            outlo = sum(data[col] < q25 - 1.5 * (q75- q25))
            outhi = sum(data[col] > q75 + 1.5 * (q75- q25))
                
        tmpdf.loc[i] = [col, datatype, recs, miss, _min, q25, q50, avg, q75, _max, stdv, uniq, outlo, outhi]
    return tmpdf



#####################################################################
# visulization function
#####################################################################

def bin_plot(out):
    '''Bar plot counts in each bin.

    Parameters
    ----------
    out : dictionary
        Bin interval as key names, row index of corresponding bin interval 
        as values.
    
    Returns
    -------
    seaborn bar plot
    '''
    df = pd.DataFrame([(k, len(out[k])) for k in out.keys()], columns=['Range', 'Number'])
    print(df)
    p = sns.barplot(x='Range', y='Number', data=df)
    p.set_xticklabels(p.get_xticklabels(), rotation=30)
    return p


# 对分箱计算的woe进行绘图
def woe_plot(fea_woe):
    '''Bar plot for woe value.

    Parameters
    ----------
    fea_woe : dictionary
        Bin interval as key names, corresponding WOE value as values.

    Returns
    -------
    seaborn bar plot
    ''' 
    for f in fea_woe:
        tmp = fea_woe[f].items()
        tmp = sorted(tmp, key=lambda x:pd.to_numeric(str(x[0]).split(',')[0]))
        x = [i[0] for i in tmp]
        y = [i[1] for i in tmp]
        print(f)
        p = sns.barplot(x=x, y=y)
        p.set_xticklabels(p.get_xticklabels(), rotation=30)
    return p


def score_ks_plot(score, label, bad=1, good=0):
    '''KS-plot

    Parameters
    ----------
    score : list or array like
        Final score value for each case.
    label : list or array like
        Label for indicating good or bad case.
    bad : str or int
        Bad case label, default is int 1.
    good : str or int
        Good case label, default is int 0.

    Returns
    -------
    KS-plot is returned with max ks-score
    '''
    label = list(label)
    items = sorted(zip(score, label), key=lambda x:x[0])
    total_good = float(label.count(good))
    total_bad = float(label.count(bad))

    step = (max(score) - min(score)) / 100

    good_list = []
    bad_list = []
    score_ticks = []

    max_dist = (0,0)
    for i in range(1, 101):
        idx = min(score) + int(i*step)
        #print(idx)
        score_ticks.append(idx)
        tmp_label = [x[1] for x in items if x[0] < idx]
        good_rate = tmp_label.count(good) / total_good
        bad_rate = tmp_label.count(bad) / total_bad

        if abs(good_rate - bad_rate) > max_dist[0]:
            max_dist = (abs(good_rate - bad_rate), i)

        good_list.append(good_rate)
        bad_list.append(bad_rate)

    ks_good = good_list[max_dist[1]]
    ks_bad = bad_list[max_dist[1]]
    ks_score = score_ticks[max_dist[1]]

    fig = plt.figure(figsize=(10,10))

    ax = fig.add_subplot(111)
    ax.plot(score_ticks, good_list, color='green', linewidth=1,
            label='Cumulative percentages of good')
    ax.plot(score_ticks, bad_list, color='red', linewidth=1,
            label='Cumulative percentages of bad')
    ax.plot([ks_score, ks_score], [ks_good, ks_bad],
            linewidth=1, linestyle='dashed', marker='o', label='KS')
    ax.annotate('(%.2f, %.2f%%, %.2f%%)\nKS=%.2f' % (ks_score,
                                                    ks_good*100,
                                                    ks_bad*100,
                                                    max_dist[0]),
                xy=(ks_score*1.1, ks_good*0.9))
    ax.legend(bbox_to_anchor=(0, -0.15),ncol=1, loc=3)
    ax.set_xlabel('Score')
    return ax



def roc_plot(y_real, y_pred, title=None):
    '''ROC-plot

    Parameters
    ----------
    y_real : list or array like
        True binary lables. Get more detial help from `sklearn.metrics.roc_curve`.
    y_pred : list or array like
        Probability estimates from model.
    title : str or int
        Bad case label, default is int 1.

    Returns
    -------
    ROC-plot is returned.
    '''
    fpr, tpr, thresh = roc_curve(y_real,  y_pred)

    sum_sensitivity_specificity_train = tpr + (1-fpr)
    best_threshold_id = np.argmax(sum_sensitivity_specificity_train)
    best_threshold = thresh[best_threshold_id]
    best_fpr = fpr[best_threshold_id]
    best_tpr = tpr[best_threshold_id]

    auc = roc_auc_score(y_real, y_pred)
    
    plt.figure(figsize=(8,8))
    plt.plot(fpr,tpr)
    plt.plot(best_fpr, best_tpr, marker='o', color='black')
    plt.text(best_fpr, 
            best_tpr, 
            s = 'Thresh:%.3f, (FPR:%.3f, TPR:%.3f)' %(thresh[best_threshold_id], best_fpr, best_tpr),
            fontsize=10)

    plt.plot([0, 1], [0, 1], color='pink', linestyle='--')
    
    plt.xlabel('False Positive Rate', fontsize=18)
    plt.ylabel('True Positive Rate', fontsize=18)
    if title:
        title_text = 'ROC curve(%s), AUC = %.4f' % (title, auc)
    else:
        title_text = 'ROC curve, AUC = %.4f' % (auc)
    plt.title(title_text, fontsize = 20, fontweight='bold')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.grid(True)
    plt.show()


# TODO
# PIS 


#####################################################################
# summary function
#####################################################################

def frequency_table(xarray, label, steps):
    '''Cases frequency table.   

    Cut score into different interval and get.

    Parameters
    ----------
    xarray : pandas.Series or numpy.array
        A final score array from score card.
    label : pandas.Series or numpy.array
        A binary array to indicate good or bad cases.
    steps : int list 
        A int list to control the interval boundary

    Returns
    -------
    pandas.DataFrame
        A frequency table group by good or bad cases. See more detials in example code.
    '''
    cols = ['Bins', 'Percent', 'Cumulative_percent', 'Counts', 'Cumulative_Counts']
    if len(set(label)) != 2:
        raise ValueError('label should be binary value.')
    cols.extend(list(set(label)))   # column names
    cols.append("%s/%s" % (cols[-1], cols[3]))
    fre_df = pd.DataFrame(columns=cols)
    total_length = float(len(xarray))
    sum_length = 0

    for (i,j) in enumerate(steps[:-1]):
        border = (steps[i], steps[i+1])
        value_idx = (xarray >= border[0]) & (xarray < border[1])
        tmp = xarray[value_idx]
        tmp_length = len(tmp)
        sum_length += tmp_length
        label_counts = label[value_idx].value_counts()
        label_counts_dict = dict(zip(label_counts.index, label_counts))

        row = ["[%s,%s)" % (border[0], border[1]),
               "%s%%" % (round(tmp_length/total_length * 100, 3)),
               "%s%%" % (round(sum_length/total_length * 100, 3)),
               tmp_length,
               sum_length,
               label_counts_dict.get(cols[-3],0),
               label_counts_dict.get(cols[-2], 0),
               "%s%%" % (round((label_counts_dict.get(cols[-2], 0)/tmp_length * 100),3) if tmp_length>0 else '-')]

        fre_df.loc[i] = row
    return fre_df


def woe_table(feature_woe, coef, slope):
    table = pd.DataFrame(columns=['Feature', 'Bin', 'WOE', 'Format', 'Score'])
    for f in coef.index:
        tmp_woe = feature_woe[f]
        bins = tmp_woe.keys()
        bins = sorted(bins,
                    key=lambda x : pd.to_numeric(str(x).split(':'))[0])
        for b in bins:
            value = slope * coef[f] * tmp_woe[b]
            row_idx = 0 if pd.isna(table.index.max()) else table.index.max()+1
            
            border = [pd.to_numeric(_i) for _i in b.split(':')]
            if border[0] == -np.inf:
                _format = "<;%s" % border[1]
            elif border[1] == np.inf:
                _format = ">=;%s" % border[0]
            else:
                _format = "<=and<;%s,%s" % (border[0], border[1])
            
            table.loc[row_idx] = [f, b, tmp_woe[b], _format, value]
    return table

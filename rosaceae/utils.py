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

from itertools import combinations
from math import log
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


# 对模型变量进行遍历分析，将结果保存在DataFrame中
def model_selecter(x_train, x_test, y_train, y_test, start=1, end=None, verbose=False):
    result_df = pd.DataFrame(columns=['Var_No', 'Vars', 'train_score', 'test_score','coef', 'inter'])
    if not end:
        end = x_train.shape[1]
    cols = x_train.columns
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

            row = 0 if pd.isna(result_df.index.max()) else result_df.index.max() + 1
            result_df.loc[row] = [n, ','.join(t), train_roc_score, test_roc_score, clf.coef_[0], clf.intercept_[0]]

    return result_df


# Use cross validation to evaluate models.
def model_selecter2(x, y, start=1, end=None, verbose=False):
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

#####################################################################
# visulization function
#####################################################################

def bin_plot(out):
    df = pd.DataFrame([(k, len(out[k])) for k in sorted(out.keys(), key=lambda x:float(str(x).split(',')[0]))], columns=['Range', 'Number'])
    print(df)
    p = sns.barplot(x='Range', y='Number', data=df)
    p.set_xticklabels(p.get_xticklabels(), rotation=30)
    return p


# 对分箱计算的woe进行绘图
def woe_plot(fea_woe):
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
    label = list(label)
    items = sorted(zip(score, label), key=lambda x:x[0])
    total_good = float(label.count(good))
    total_bad = float(label.count(bad))

    step = int(len(score) / 100)

    good_list = []
    bad_list = []
    score_ticks = []

    max_dist = (0,0)
    for i in range(1, 101):
        idx = int(i*step)
        #print(idx)
        score_ticks.append(items[idx][0])
        tmp_label = [x[1] for x in items[0:idx]]
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

#####################################################################
# summary function
#####################################################################
# TODO: feature importance and IV

def frequent_table(xarray, label, steps):
    cols = ['Bins', 'Percent', 'Cumulative_percent', 'Counts', 'Cumulative_Counts']
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

        row = [str(border),
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

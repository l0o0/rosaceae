# -*- coding: utf-8 -*-
"""
rosaceae.utils
~~~~~~~~~~~~~~

This module provides utility functions that are used within Rosaceae.
Including visulization and summary functions.
"""

import seaborn as sns

from itertools import combinations


# 对模型变量进行遍历分析，将结果保存在DataFrame中
def model_selecter(x_train, x_test, y_train, y_test, start=1, end=None, verbose=False):
    result_df = pd.DataFrame(columns=['Var_No', 'Vars', 'train_score', 'test_score','coef', 'inter'])
    if not end:
        end = x_train.shape[1]
    cols = x_train.columns
    for n in range(start, end+1):
        cols_try = combinations(cols, n)

        if not verbose:
            print n

        for t in cols_try:
            tmp_train = x_train.loc[:, x_train.columns.isin(t)]
            tmp_test = x_test.loc[:, x_test.columns.isin(t)]
            clf = LogisticRegression(random_state=0)
            clf.fit(tmp_train, y_train)
            train_roc_score = roc_auc_score(y_train, clf.decision_function(tmp_train))
            test_roc_score = roc_auc_score(y_test, clf.decision_function(tmp_test))
            if verbose:
                print "%s\t%s\t%s\t%s" % (n, ','.join(t), train_roc_score, test_roc_score)

            row = 0 if pd.isna(result_df.index.max()) else result_df.index.max() + 1
            result_df.loc[row] = [n, ','.join(t), train_roc_score, test_roc_score, clf.coef_, clf.intercept_]

    return result_df


#####################################################################
# visulization function
#####################################################################
# TODO(l0o0): KS plot is needed.

def bin_plot(out):
    df = pd.DataFrame([(k, len(out[k])) for k in sorted(out.keys(), key=lambda x:float(str(x).split(',')[0]))], columns=['Range', 'Number'])
    print df
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
        print f
        p = sns.barplot(x=x, y=y)
        p.set_xticklabels(p.get_xticklabels(), rotation=30)
    return p


#####################################################################
# summary function
#####################################################################
# TODO: feature importance and IV 

def frequent_table(xarray, label, steps):
    cols = ['Bins', 'Percent', 'Cumulative_percent', 'Counts', 'Cumulative_Counts']
    cols.extend(list(set(label)))   # column names
    fre_df = pd.DataFrame(columns=cols)

    total_length = float(len(xarray))
    sum_length = 0

    for i,j in enumerate(steps[:-1]):
        border = (steps[i], steps[i+1])
        value_idx = (xarray >= border[0]) & (xarray < border[1])
        tmp = xarray[value_idx]
        tmp_length = len(tmp)
        sum_length += tmp_length
        label_counts = label[value_idx].value_counts()
        label_counts_dict = dict(zip(label_counts.index, label_counts))
        #print label_counts_dict
        row = [str(border),
               "%f%%" % (tmp_length/total_length * 100),
               "%f%%" % (sum_length/total_length * 100),
               tmp_length,
               sum_length,
               label_counts_dict.get(cols[-2],0),
               label_counts_dict.get(cols[-1], 0)]

        fre_df.loc[i] = row
    return fre_df

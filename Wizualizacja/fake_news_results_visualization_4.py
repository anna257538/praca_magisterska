import datetime
from distutils import errors
import gc
import json
import os
import re
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             confusion_matrix, precision_recall_fscore_support)
from scipy.stats import shapiro, f, ttest_ind
import dataframe_image as dfi

warnings.simplefilter(action='ignore', category=FutureWarning)

vects_new_names = {
    'TfidfVectorizer0': 'TfidfWord_1',
    'TfidfVectorizer1': 'TfidfWord_2',
    'TfidfVectorizer2': 'TfidfWord_1-2',
    'TfidfVectorizer3': 'TfidfWord_1-3',
    'TfidfVectorizer4': 'TfidfWord_1-4',
    'TfidfVectorizer5': 'TfidfChar_1',
    'TfidfVectorizer6': 'TfidfChar_2',
    'TfidfVectorizer7': 'TfidfChar_1-2',
    'TfidfVectorizer8': 'TfidfChar_1-3',
    'TfidfVectorizer9': 'TfidfChar_1-4',
    'CountVectorizer10': 'CountWord_1',
    'CountVectorizer11': 'CountWord_2',
    'CountVectorizer12': 'CountWord_1-2',
    'CountVectorizer13': 'CountWord_1-3',
    'CountVectorizer14': 'CountWord_1-4',
    'CountVectorizer15': 'CountChar_1',
    'CountVectorizer16': 'CountChar_2',
    'CountVectorizer17': 'CountChar_1-2',
    'CountVectorizer18': 'CountChar_1-3',
    'CountVectorizer19': 'CountChar_1-4',
    'NoneType20': "Custom"
}

cfls_new_names = {
    'SVC0': "SvmRbf",
    'SVC1': "SvmLinear",
    'SVC2': "SvmSigmoid",
    'DecisionTreeClassifier3': 'DecisionTreeClassifier1',
    'DecisionTreeClassifier4': 'DecisionTreeClassifier2',
    'DecisionTreeClassifier5': 'DecisionTreeClassifier3',
    'RandomForestClassifier6': 'RandomForestClassifier1',
    'RandomForestClassifier7': 'RandomForestClassifier2',
    'RandomForestClassifier8': 'RandomForestClassifier3',
    'MultinomialNB12': 'MultinomialNB',
    'ComplementNB13': 'ComplementNB',
    'BernoulliNB14': 'BernoulliNB'
}


def create_table(df, param, ds_name):
    # print(df[param])
    new_df = df[param].reindex(columns=[
        'SVC0',
        'SVC1',
        'SVC2',
        'DecisionTreeClassifier3',
        'DecisionTreeClassifier4',
        'DecisionTreeClassifier5',
        'RandomForestClassifier6',
        'RandomForestClassifier7',
        'RandomForestClassifier8',
        'MultinomialNB12',
        'ComplementNB13',
        'BernoulliNB14',
        'DummyClassifier15'
    ])

    vects = [
        'TfidfVectorizer0',
        'TfidfVectorizer1',
        'TfidfVectorizer2',
        'TfidfVectorizer3',
        'TfidfVectorizer4',
        'TfidfVectorizer5',
        'TfidfVectorizer6',
        'TfidfVectorizer7',
        'TfidfVectorizer8',
        'TfidfVectorizer9',
        'CountVectorizer10',
        'CountVectorizer11',
        'CountVectorizer12',
        'CountVectorizer13',
        'CountVectorizer14',
        'CountVectorizer15',
        'CountVectorizer16',
        'CountVectorizer17',
        'CountVectorizer18',
        'CountVectorizer19',
        'NoneType20'
    ]
    new_df = new_df.reindex(vects)

    plt.figure(figsize=(16, 20))
    plt.title("{} - {}".format(ds_name, param))
    sns.heatmap(new_df, annot=True)
    file_name = "wykresiki/{}-{}-{}.png".format(
        ds_name, param, datetime.datetime.now().timestamp())
    if os.path.exists(file_name):
        os.remove(file_name)
    plt.savefig(file_name)


def read_from_one_file(ds_name):
    filename = 'wyniki_{}.txt'.format(ds_name)
    with open(filename, 'r') as json_file:
        df = pd.read_json(json.load(json_file))
    return df


def read_from_all_files_and_write_to_one(ds_name):
    df = pd.DataFrame(
        columns=['classifier', "vectorizer", 'accuracy', 'balanced accuracy', 'precision', 'recall', 'F-measure'])
    path = "wyniki/{}".format(ds_name)
    for filename in os.listdir(path):
        if re.match("{}_.*\.txt".format(ds_name), filename):
            with open(os.path.join(path, filename), 'r') as json_file:
                data = json.load(json_file)
                clf = data['classifier'][0]
                vect = data["vectorizer"][0]
                all_ys = data["y_tests"]
                all_predicted = data["predicted"]

                accs = []
                balanced_accs = []
                precs = []
                recalls = []
                fms = []

                for i in range(len(all_ys)):
                    p, r, f, s = precision_recall_fscore_support(
                        all_ys[i], all_predicted[i], zero_division=0, average='weighted')

                    df = df.append({
                        'classifier': clf,
                        'vectorizer': vect,
                        'accuracy': accuracy_score(all_ys[i], all_predicted[i]),
                        'balanced accuracy': balanced_accuracy_score(all_ys[i], all_predicted[i]),
                        'precision': p,
                        'recall': r,
                        'F-measure': f
                    }, ignore_index=True)

    with open('wyniki_{}.txt'.format(ds_name, datetime.datetime.now().timestamp()), 'w') as file:
        json.dump(df.to_json(), file)


def f_test(x, y):
    x = np.array(x)
    y = np.array(y)
    v1 = np.var(x, ddof=1)
    v2 = np.var(y, ddof=1)
    fs = v1 / v2
    dfn = x.size-1
    dfd = y.size-1
    p = 1-f.cdf(fs, dfn, dfd)
    return fs, p


def compare_results(df, clf1, clf2, vect1, vect2, param):
    if clf1 != None:
        a1 = df.loc[df['classifier'] == clf1]
        a2 = df.loc[df['classifier'] == clf2]
    else:
        a1 = df
        a2 = df

    if vect1 != None:
        a1 = a1.loc[a1['vectorizer'] == vect1]
        a2 = a2.loc[a2['vectorizer'] == vect2]

    a1 = a1[param]
    a2 = a2[param]

    return np.mean(a1), np.mean(a2), np.var(a1, ddof=1), np.var(a2, ddof=1), f_test(a1, a2)[0], f_test(a1, a2)[1]


def compare_clfs_to_dummy(df, clfs):
    res = pd.DataFrame(
        columns=['clf1', 'clf2', 'mean1', "mean2", 'var1', 'var2', 'f', 'p-value'])

    for indx, g in enumerate([item for sublist in clfs for item in sublist]):
            # for i in range(len(g)):
            # for j in range(i+1, len(g)):
        mean1, mean2, var1, var2, f, t = compare_results(
            df, g, 'DummyClassifier15', None, None, 'accuracy')
        res = res.append({
            'clf1': g,
            'clf2': "DummyClassifier15",
            'mean1': mean1,
            "mean2": mean2,
            'var1': var1,
            'var2': var2,
            'f': f,
            'p-value': t
        }, ignore_index=True)

    # print(res)

    df_styled = pd.pivot_table(res, values=['p-value'],
                               index=["clf1"], columns=["clf2"])
    df_styled = df_styled.style.applymap(
        lambda v: 'color:green;' if (v < 0.05) else None).format(formatter="{:.4g}")
    dfi.export(df_styled, "clf_f.png".format(indx))
    # df_styled = pd.pivot_table(res, values=['t-student'],
    #                            index=["clf1"], columns=["clf2"])
    # df_styled = df_styled.style.applymap(
    #     lambda v: 'color:green;' if (v < 0.05) else None).format(formatter="{:.4g}")
    # dfi.export(df_styled, "clf_t.png".format(indx))


def compare_pairs(df, vects, clfs, ds_name):
    df = df[df.classifier != "DummyClassifier15"]

    res = pd.DataFrame(
        columns=['pair1', 'pair22', 'mean1', "mean2", 'p-value'])

    for indxC1, clf1 in enumerate(clfs):
        for indxV1, vec1 in enumerate(vects):
            for indxC2, clf2 in enumerate(clfs):
                for indxV2, vec2 in enumerate(vects):
                    mean1, mean2, var1, var2, f, t = compare_results(
                        df, clf1, clf2, vec1, vec2, 'accuracy')
                    res = res.append({
                        'pair1': (cfls_new_names[clf1], vects_new_names[vec1]),
                        'pair2': (cfls_new_names[clf2], vects_new_names[vec2]),
                        'mean1': mean1,
                        "mean2": mean2,
                        'var1': var1,
                        'var2': var2,
                        'f': f,
                        'p-value': t
                    }, ignore_index=True)

            # print(res)

    df_styled2 = pd.pivot_table(res, values=['p-value'],
                                index=["pair1"], columns=["pair2"])

    df_styled2 = df_styled2['p-value']

    # print(df_styled2)
    df_styled2 = df_styled2.style.applymap(
        lambda v: 'color:green;' if (v < 0.05) else None).format(formatter="{:.4g}")
    dfi.export(df_styled2, "pairs_p_{}.png".format(ds_name))
            # dfi.export(df_styled2, "vects_p_{}_{}_all.png".format(indx, ds_name))


def compare_clfs(df, clfs, ds_name):
    df = df[df.classifier != "DummyClassifier15"]

    for indx, g in enumerate(clfs):

        res = pd.DataFrame(
            columns=['clf1', 'clf2', 'mean1', "mean2", 'var1', 'var2', 'f', 'p-value'])
        for i in range(len(g)):
            for j in range(len(g)):
                mean1, mean2, var1, var2, f, t = compare_results(
                    df, g[i], g[j], None, None, 'accuracy')
                res = res.append({
                    'clf1': g[i],
                    'clf2': g[j],
                    'mean1': mean1,
                    "mean2": mean2,
                    'var1': var1,
                    'var2': var2,
                    'f': f,
                    'p-value': t
                }, ignore_index=True)

        # df_styled = pd.pivot_table(res, values=['f'],
        #                            index=["clf2"], columns=["clf1"])
        # df_styled = df_styled.style.applymap(
        #     lambda v: 'color:green;' if (v < 0.05) else None).format(formatter="{:.4g}")
        # dfi.export(df_styled, "clfs_f_{}.png".format(indx))

        df_styled = pd.pivot_table(res, values=['p-value'],
                                   index=["clf2"], columns=["clf1"])

        print(df_styled)
        df_styled = df_styled['p-value'].rename(
            index=cfls_new_names, level=0)
        df_styled = df_styled.rename(columns=cfls_new_names, level='clf1')

        df_styled = df_styled.style.applymap(
            lambda v: 'color:green;' if (v < 0.05) else None).format(formatter="{:.4g}")
        dfi.export(df_styled, "clfs_p_{}_{}_all.png".format(indx, ds_name))


def main(argv):
    # datasets = ["kaggle", "isot"]
    datasets = ["isot"]
    # prfs = ['precision', 'recall', 'F-measure', 'support']

    for ds_name in datasets:
        # read_from_all_files_and_write_to_one(ds_name)
        df = read_from_one_file(ds_name)

        df = df[['classifier', 'vectorizer', 'accuracy']]

        # print(df)
        # print(df.shape)

        clfs = np.unique(df['classifier'])
        # vects = np.unique(df['vectorizer'])
        # print(clfs)
        # print(vects)

        # for c in clfs:
        # for v in vects:

        # pd.options.display.float_format = '{:,}'.format
        # print(str(df.loc[df['classifier'] == 'SVC2']
        #           .loc[df['vectorizer'] == 'TfidfVectorizer0']['balanced accuracy']).replace('.', ','))
        # print(str(df.loc[df['classifier'] == 'SVC0']
        #           .loc[df['vectorizer'] == 'TfidfVectorizer0']['balanced accuracy']).replace('.', ','))

        # print(f_test(df.loc[df['classifier'] == 'SVC2'].loc[df['vectorizer'] == 'TfidfVectorizer0']['balanced accuracy'],
        #              df.loc[df['classifier'] == 'SVC0'].loc[df['vectorizer'] == 'TfidfVectorizer0']['balanced accuracy']))

        vects = [
            ['TfidfVectorizer0',
             'TfidfVectorizer1',
             'TfidfVectorizer2',
             'TfidfVectorizer3',
             'TfidfVectorizer4'],
            ['TfidfVectorizer5',
             'TfidfVectorizer6',
             'TfidfVectorizer7',
             'TfidfVectorizer8',
             'TfidfVectorizer9'],
            ['CountVectorizer10',
             'CountVectorizer11',
             'CountVectorizer12',
             'CountVectorizer13',
             'CountVectorizer14'],
            ['CountVectorizer15',
             'CountVectorizer16',
             'CountVectorizer17',
             'CountVectorizer18',
             'CountVectorizer19'],
            ['NoneType20']]

        clfs = [
            ['SVC0',
             'SVC1',
             'SVC2'],
            ['DecisionTreeClassifier3',
             'DecisionTreeClassifier4',
             'DecisionTreeClassifier5'],
            ['RandomForestClassifier6',
             'RandomForestClassifier7',
             'RandomForestClassifier8'],
            ['MultinomialNB12',
             'ComplementNB13',
             'BernoulliNB14']
        ]

        # compare_vects(df, vects, ds_name)
        # compare_clfs_to_dummy(df, clfs)
        # compare_clfs(df, clfs, ds_name)

        # df = df[df.classifier != "DummyClassifier15"]
        # res = pd.DataFrame(
        #     columns=['Zbiór cech', 'Średnia', 'Wariancja'])

        # for indx, g in enumerate([item for sublist in vects for item in sublist]):
        #     a1 = df.loc[df['vectorizer'] == g]['accuracy']
        #     res = res.append({
        #         'Zbiór cech': vects_new_names[g],
        #         'Średnia': np.mean(a1),
        #         'Wariancja': np.var(a1, ddof=1)
        #     }, ignore_index=True)

        # compare_vects(df, [['TfidfVectorizer2', 'TfidfVectorizer8', 'CountVectorizer12', 'CountVectorizer19', 'NoneType20']], ds_name)

        # plt.figure(figsize=(15,15))
        # b = plt.bar(res['Zbiór cech'], res['Średnia'])
        # plt.xticks(
        #     rotation=45, ha='right'
        # )
        # plt.bar_label(b, fmt='%.3f')
        # plt.savefig('vects_mean_{}.png'.format(ds_name))

        # df_styled = res.style.applymap(
        #         lambda v: 'color:green;' if (v < 0.05) else None).format(formatter="{:.4g}")
        # dfi.export(res, "vects_mean_var_{}.png".format(ds_name))

        compare_pairs(df, ['TfidfVectorizer1', 'TfidfVectorizer6', 'CountVectorizer13', 'CountVectorizer16'],
                      ['SVC1', 'DecisionTreeClassifier4', 'RandomForestClassifier6', 'BernoulliNB14'], ds_name)

        # res = pd.DataFrame(
        #     columns=['Klasyfikator', 'Zbiór cech', 'Średnia', 'Wariancja'])

        # for indx, g in enumerate(['SVC1', 'DecisionTreeClassifier4', 'RandomForestClassifier6', 'BernoulliNB14']):
        #     for indx, h in enumerate(['TfidfVectorizer1', 'TfidfVectorizer6', 'CountVectorizer13', 'CountVectorizer16']):
        #         a1 = df.loc[df['classifier'] == g]
        #         a1 = a1.loc[df['vectorizer'] == h]['accuracy']
        #         print(a1.shape)
        #         res = res.append({
        #             'Klasyfikator': cfls_new_names[g],
        #             'Zbiór cech': vects_new_names[h],
        #             'Średnia': np.mean(a1),
        #             'Wariancja': np.var(a1, ddof=1)
        #         }, ignore_index=True)
        
        # print(res)

        # plt.figure(figsize=(15, 15))
        # b = plt.bar(res['Klasyfikator'], res['Średnia'])
        # plt.xticks(
        #     rotation=45, ha='right'
        # )
        # plt.bar_label(b, fmt='%.3f')
        # plt.savefig('clfs_mean_{}.png'.format(ds_name))

    # print(f_test(df.loc[df['classifier'] == 'SVC2'].loc[df['vectorizer'] == 'CountVectorizer19']['balanced accuracy'],
    #  df.loc[df['classifier'] == 'SVC0'].loc[df['vectorizer'] == 'CountVectorizer19']['balanced accuracy']))
    # if v != "NoneType20":
    # print(c)
    # print(v)
    # print(df.loc[df['vectorizer'] == v].loc[df['classifier'] == c]['balanced accuracy'])
    # print(shapiro(df.loc[df['vectorizer'] == v].loc[df['classifier'] == c]['balanced accuracy']))
    # print()

    # create_table(df, 'accuracy', ds_name)
    # create_table(df, 'balanced accuracy', ds_name)
    # create_table(df, 'precision', ds_name)
    # create_table(df, 'recall', ds_name)
    # create_table(df, 'F-measure', ds_name)


if __name__ == "__main__":
    main(sys.argv[1:])

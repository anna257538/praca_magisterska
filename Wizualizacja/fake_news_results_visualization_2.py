import json
import os
import sys
import warnings

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

warnings.simplefilter(action='ignore', category=FutureWarning)


def read_from_one_file(ds_name):
    filename = 'wyniki_{}.txt'.format(ds_name)
    with open(filename, 'r') as json_file:
        df = pd.read_json(json.load(json_file))
    return df


def create_table(df, param, ds_name):
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
        'BernoulliNB14'
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

    new_df = new_df.rename(columns={
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
    }, level=0)

    new_df = new_df.rename(index={
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
    }, level=0)


    plt.figure(figsize=(16, 24))


    # plt.figure(figsize=(20, 24))
    # plt.title("{} - {}".format(ds_name, param))

    sns.set(font_scale=2)

    sns.heatmap(new_df, annot=True)
    plt.xticks(
        rotation=45
    )
    plt.tick_params(axis='both', labelsize=18)
    plt.xlabel('Klasyfikator', fontsize=18)
    plt.ylabel("Metoda ekstrakcji", fontsize=18)
    file_name = "wykresiki/{}-{}_2.png".format(
        ds_name, param)
    if os.path.exists(file_name):
        os.remove(file_name)
    plt.savefig(file_name)


def main(argv):
    datasets = ["kaggle", "isot"]
    df = pd.DataFrame(
        columns=['classifier', "vectorizer", 'accuracy', 'balanced accuracy', 'precision', 'recall', 'F-measure'])
    for ds_name in datasets:
        # read_from_all_files_and_write_to_one(ds_name)
        df = read_from_one_file(ds_name)
        df = df.rename(columns={
            'classifier': 'Klasyfikator',
            'vectorizer': "Metoda ekstrakcji"})

    # for ds_name in datasets:
    #     path = "wyniki/{}".format(ds_name)
    #     for filename in os.listdir(path):
    #         if re.match("{}_.*\.txt".format(ds_name), filename):
    #             with open(os.path.join(path, filename), 'r') as json_file:
    #                 data = json.load(json_file)
    #                 clf = data['classifier'][0]
    #                 vect = data["vectorizer"][0]
    #                 all_ys = data["y_tests"]
    #                 all_predicted = data["predicted"]

    #                 accs = []
    #                 balanced_accs = []
    #                 precs = []
    #                 recalls = []
    #                 fms = []

    #                 for i in range(len(all_ys)):
    #                     accs.append(accuracy_score(
    #                         all_ys[i], all_predicted[i]))
    #                     balanced_accs.append(balanced_accuracy_score(
    #                         all_ys[i], all_predicted[i]))
    #                     p, r, f, s = precision_recall_fscore_support(
    #                         all_ys[i], all_predicted[i], zero_division=0)
    #                     precs.append(p)
    #                     recalls.append(r)
    #                     fms.append(f)

        # df = df.append({
        #     'classifier': clf,
        #     'vectorizer': vect,
        #     'accuracy': np.mean(accs),
        #     'balanced accuracy': np.mean(balanced_accs),
        #     'precision': np.mean(precs),
        #     'recall': np.mean(recalls),
        #     'F-measure': np.mean(fms)
        # }, ignore_index=True)

        df = pd.pivot_table(df, values=['accuracy', 'precision', 'recall', 'F-measure'],
                            index=["Metoda ekstrakcji"], columns=["Klasyfikator"])
        create_table(df, 'accuracy', ds_name)
        create_table(df, 'precision', ds_name)
        create_table(df, 'recall', ds_name)
        create_table(df, 'F-measure', ds_name)


if __name__ == "__main__":
    main(sys.argv[1:])

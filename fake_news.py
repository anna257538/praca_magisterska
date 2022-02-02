import argparse
import datetime
import gc
import json
import re
import sys
import warnings

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.naive_bayes import BernoulliNB, ComplementNB, MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

warnings.simplefilter(action='ignore', category=FutureWarning)


# root_dir = "Datasets\\"
root_dir = "Datasets/"



def verifyClassifierWithParams(dataset, classifier, vectorizer, use_counts=False):
    # utworzenie klasyfikatora z zadanymi parametrami oraz podzielenie zbioru danych na podzbiory
    dataset.reset_index(drop=True, inplace=True)

    texts = dataset['text'].values
    classes = dataset['label'].values
    counts = dataset.drop(columns=['text', 'label'])

    cross_validator = RepeatedStratifiedKFold(
        n_splits=2, n_repeats=5, random_state=7890)

    y_tests = []
    predicted = []

    # uczenie i testowanie klasyfikatora utworzonymi podzbiorami
    for train_index, test_index in cross_validator.split(texts, classes):
        X_train, X_test = texts[train_index], texts[test_index]
        y_train, y_test = classes[train_index], classes[test_index]

        # przesuwanie wartości powyżej 0
        counts['F-K grade'] = counts['F-K grade'] + 4

        c_train, c_test = counts.loc[train_index, :], counts.loc[test_index, :]

        if vectorizer == None:
            X_train = c_train
            X_test = c_test
        else:
            X_train = vectorizer.fit_transform(X_train)
            X_test = vectorizer.transform(X_test)

        if use_counts and vectorizer != None:
            X_train, X_test = add_counts(X_train, X_test, c_train, c_test)
        else:
            X_train = sparse.csr_matrix(X_train)
            X_test = sparse.csr_matrix(X_test)
            
        # wymagane do uruchomienia testów dla SVC z liniowym jądrem
        # scaling = MinMaxScaler(feature_range=(-1, 1)).fit(X_train)
        # X_train = scaling.transform(X_train)
        # X_test = scaling.transform(X_test)

        classifier.fit(X_train, y_train)
        predict = classifier.predict(X_test)

        y_tests.append(y_test.tolist())
        predicted.append(predict.tolist())

    return y_tests, predicted


def add_counts(X_train, X_test, c_train, c_test):
    c_train = sparse.csr_matrix(c_train)
    c_test = sparse.csr_matrix(c_test)

    X_train = sparse.hstack((X_train, c_train))
    X_test = sparse.hstack((X_test, c_test))

    return sparse.csr_matrix(X_train), sparse.csr_matrix(X_test)


def preproc(s):
    return re.sub(r'[0-9][0-9.,-]*', ' NUMBERSPECIALTOKEN ', s).lower()


def main(datafile, clsf_idx, vect_idx, use_counts):
    dataset = pd.read_csv(root_dir + datafile, dtype={'label': str})

    classifiers = [
        SVC(),
        SVC(kernel='linear'),
        SVC(kernel='sigmoid'),
        DecisionTreeClassifier(),
        DecisionTreeClassifier(
            criterion="gini", max_depth=30, max_leaf_nodes=200),
        DecisionTreeClassifier(criterion="entropy",
                               max_depth=50, max_leaf_nodes=1000),
        RandomForestClassifier(),
        RandomForestClassifier(1000, criterion="gini", min_samples_split=20),
        # RandomForestClassifier(criterion='entropy', min_samples_split=50), # wymagana zamiana z kolejną linijką, gdy całkowita liczba cech <75
        RandomForestClassifier(criterion='entropy', min_samples_split=50, max_features=75),
        MultinomialNB(),
        ComplementNB(),
        BernoulliNB(),
        DummyClassifier(strategy='stratified')]

    vectorizers = [
        TfidfVectorizer(ngram_range=(1, 1),
                        stop_words='english', preprocessor=preproc),
        TfidfVectorizer(ngram_range=(2, 2),
                        stop_words='english', preprocessor=preproc),
        TfidfVectorizer(ngram_range=(1, 2),
                        stop_words='english', preprocessor=preproc),
        TfidfVectorizer(ngram_range=(1, 3),
                        stop_words='english', preprocessor=preproc),
        TfidfVectorizer(ngram_range=(1, 4),
                        stop_words='english', preprocessor=preproc),
        TfidfVectorizer(ngram_range=(1, 1), stop_words='english',
                        preprocessor=preproc, analyzer='char'),
        TfidfVectorizer(ngram_range=(2, 2), stop_words='english',
                        preprocessor=preproc, analyzer='char'),
        TfidfVectorizer(ngram_range=(1, 2), stop_words='english',
                        preprocessor=preproc, analyzer='char'),
        TfidfVectorizer(ngram_range=(1, 3), stop_words='english',
                        preprocessor=preproc, analyzer='char'),
        TfidfVectorizer(ngram_range=(1, 4), stop_words='english',
                        preprocessor=preproc, analyzer='char'),
        CountVectorizer(ngram_range=(1, 1),
                        stop_words='english', preprocessor=preproc),
        CountVectorizer(ngram_range=(2, 2),
                        stop_words='english', preprocessor=preproc),
        CountVectorizer(ngram_range=(1, 2),
                        stop_words='english', preprocessor=preproc),
        CountVectorizer(ngram_range=(1, 3),
                        stop_words='english', preprocessor=preproc),
        CountVectorizer(ngram_range=(1, 4),
                        stop_words='english', preprocessor=preproc),
        CountVectorizer(ngram_range=(1, 1), stop_words='english',
                        preprocessor=preproc, analyzer='char'),
        CountVectorizer(ngram_range=(2, 2), stop_words='english',
                        preprocessor=preproc, analyzer='char'),
        CountVectorizer(ngram_range=(1, 2), stop_words='english',
                        preprocessor=preproc, analyzer='char'),
        CountVectorizer(ngram_range=(1, 3), stop_words='english',
                        preprocessor=preproc, analyzer='char'),
        CountVectorizer(ngram_range=(1, 4), stop_words='english',
                        preprocessor=preproc, analyzer='char'),
        None]

    results = {
        'classifier': [],
        'classifier_params': [],
        'vectorizer': [],
        'vectorizer_params': [],
        'y_tests': [],
        'predicted': []
    }

    clsf = classifiers[clsf_idx]
    vect = vectorizers[vect_idx]

    y_tests, predicted = verifyClassifierWithParams(
        dataset, clsf, vect, use_counts)

    results['classifier'].extend([clsf.__class__.__name__ + str(clsf_idx)])
    results['vectorizer'].extend([vect.__class__.__name__ + str(vect_idx)])

    results['classifier_params'].append(str(clsf.get_params()))
    results['vectorizer_params'].append(
        str(vect.get_params() if vect != None else None))

    results['y_tests'] = y_tests
    results['predicted'] = predicted

    with open('{}-{}-{}-{}-{}.txt'.format(datafile, clsf_idx, vect_idx, use_counts, datetime.datetime.now().timestamp()), 'w') as file:
        json.dump(results, file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("datafile")
    parser.add_argument("clsf_idx", type=int)
    parser.add_argument("vect_idx", type=int)
    parser.add_argument("--use_counts", action="store_true")
    args = parser.parse_args()
    main(args.datafile, args.clsf_idx, args.vect_idx, args.use_counts)
